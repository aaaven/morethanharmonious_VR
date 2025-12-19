from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL



class AugEmbedding(nn.Module):

    def __init__(self, add_time_proj, add_embedding):
        super().__init__()
        self.add_time_proj = add_time_proj
        self.add_embedding = add_embedding
    
    def forward(self, time_ids, text_embeds):
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))  # 2 1536
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        aug_emb = self.add_embedding(add_embeds)
        return aug_emb

class UNet2D(nn.Module):

    def __init__(self, conv_in, down_blocks, mid_block, up_blocks, conv_norm_out, conv_act, conv_out):
        super().__init__()
        self.conv_in = conv_in
        self.down_blocks = down_blocks
        self.mid_block = mid_block
        self.up_blocks = up_blocks
        self.conv_norm_out = conv_norm_out
        self.conv_act = conv_act
        self.conv_out = conv_out

    def forward(self, sample, emb, encoder_hidden_states):
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if getattr(downsample_block, "has_cross_attention", False):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if getattr(self.mid_block, "has_cross_attention", False):
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                )

        # 5. up
        for upsample_block in self.up_blocks:

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if getattr(upsample_block, "has_cross_attention", False):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

class Scheduler(nn.Module):

    def __init__(self, time_proj, time_embedding):
        super().__init__()
        self.time_proj = time_proj
        self.time_embedding = time_embedding
    
    def set_inference_params(self, num_inference_steps, strength, guidance_scale):

        num_train_timesteps = 1000
        self.num_inference_steps = num_inference_steps
        beta_start = 0.00085
        beta_end = 0.012

        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        step_ratio = num_train_timesteps / self.num_inference_steps
        timesteps = (
            (np.arange(num_train_timesteps - 1, -1, -step_ratio))
            .round()
            .astype(np.float32)
        )
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.tensor(timesteps)

        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        self.timesteps = self.timesteps[t_start:]
        self.sigmas = self.sigmas[t_start:]
        self.sigmas_from = self.sigmas[:-1]
        self.sigmas_to = self.sigmas[1:]
        self.sigmas_up = (
            self.sigmas_to**2
            * (self.sigmas_from**2 - self.sigmas_to**2)
            / self.sigmas_from**2
        ) ** 0.5
        self.sigmas_down = (self.sigmas_to**2 - self.sigmas_up**2) ** 0.5
        self.dt = self.sigmas_down - self.sigmas_from
        
        self.timeembs = self.time_embedding(self.time_proj(self.timesteps))
        self.guidance_scale = guidance_scale
    
class Pipeline(nn.Module):

    unet: UNet2D
    vae: AutoencoderKL

    def __init__(self, unet, vae):
        super().__init__()
        self.unet = unet
        self.vae = vae

    def set_scheduler(self, scheduler: Scheduler):
        self.sigmas = scheduler.sigmas
        self.timeembs = scheduler.timeembs
        self.sigmas_up = scheduler.sigmas_up
        self.dt = scheduler.dt
        self.guidance_scale = scheduler.guidance_scale

    def to(self, device, dtype=None):
        self.sigmas = self.sigmas.to(device, dtype)
        self.timeembs = self.timeembs.to(device, dtype)
        self.sigmas_up = self.sigmas_up.to(device, dtype)
        self.dt = self.dt.to(device, dtype)
        return super().to(device, dtype)

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        aug_emb: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor,
    ):
        # 6. Prepare latent variables
        image_latents = self.vae.encode(image * 2 - 1).latent_dist.sample()  # (1, 4, 64, 64)
        image_latents *= 0.13025
        image_latents = image_latents.to(dtype=dtype)
        noise = torch.randn_like(image_latents)
        latents = image_latents + noise * self.sigmas[0]

        # 7. Prepare mask latent variables
        mask = torch.nn.functional.interpolate(mask, size=(64, 64))

        # 11. Denoising loop
        num_timesteps = len(self.timeembs)

        for i in range(num_timesteps):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input *= (self.sigmas[i] ** 2 + 1) ** -0.5
            # predict the noise residual
            noise_pred = self.unet(
                sample=latent_model_input,
                emb=torch.stack([self.timeembs[i]] * 2) + aug_emb,
                encoder_hidden_states=prompt_embeds,
            )

            noise_pred = torch.lerp(noise_pred[0], noise_pred[1], self.guidance_scale)

            noise = torch.randn_like(noise_pred)
            latents += noise_pred * self.dt[i] + noise * self.sigmas_up[i]

            if i < num_timesteps - 1:
                # init_latents_proper = self.scheduler.add_noise(image_latents, noise)
                init_latents_proper = image_latents + noise * self.sigmas[i + 1]
            else:
                init_latents_proper = image_latents

            torch.lerp(init_latents_proper, latents, mask, out=latents)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        latents /= 0.13025

        # latents = latents.to(dtype=torch.float32)
        image = self.vae.decoder(self.vae.post_quant_conv(latents))

        image = (image / 2 + 0.5).clamp(0, 1)

        return image

class PromptEncoder:

    def __init__(self, tokenizer, tokenizer_2, text_encoder, text_encoder_2):
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

    def __call__(self, prompt: str):
        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids

            prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.view(bs_embed, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(bs_embed, -1)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

        prompt_embeds = prompt_embeds.view(2, bs_embed, seq_len, -1).permute(1, 0, 2, 3)
        pooled_prompt_embeds = pooled_prompt_embeds.view(2, bs_embed, -1).permute(
            1, 0, 2
        )

        return (prompt_embeds, pooled_prompt_embeds)

class TimeEmbdding(nn.Module):
        
    def __init__(self, time_proj, time_embedding):
        self.time_proj = time_proj
        self.time_embedding = time_embedding

    def forward(self, timesteps):
        return self.time_embedding(self.time_proj(timesteps))


pipe = StableDiffusionXLInpaintPipeline.from_pretrained("sdxl-turbo")
pipe.disable_xformers_memory_efficient_attention()


pipe_reduced = Pipeline(
    UNet2D(
        conv_in=pipe.unet.conv_in,              # Conv2d
        down_blocks=pipe.unet.down_blocks,      # DownBlock2D, CrossAttnDownBlock2D, CrossAttnDownBlock2D
        mid_block=pipe.unet.mid_block,          # UNetMidBlock2DCrossAttn
        up_blocks=pipe.unet.up_blocks,          # CrossAttnUpBlock2D, CrossAttnUpBlock2D, UpBlock2D
        conv_norm_out=pipe.unet.conv_norm_out,  # GroupNorm
        conv_act=pipe.unet.conv_act,            # SiLU
        conv_out=pipe.unet.conv_out,            # Conv2d
    ),
    pipe.vae,
)

prompt_encoder = PromptEncoder(
    pipe.tokenizer, pipe.tokenizer_2, pipe.text_encoder, pipe.text_encoder_2
)
aug_embeder = AugEmbedding(
    pipe.unet.add_time_proj,
    pipe.unet.add_embedding
)

scheduler = Scheduler(
    pipe.unet.time_proj,
    pipe.unet.time_embedding
)

strength = 0.5
guidance_scale = 1.25
num_inference_steps = 6

scheduler.set_inference_params(num_inference_steps, strength, guidance_scale)
pipe_reduced.set_scheduler(scheduler)


prompts = [
    "Abandoned Chernobyl amusement park with a rusted Ferris wheel against a radioactive sky.",
"Exxon Valdez oil spill covering the waters of Prince William Sound, with distressed wildlife.",
"Dense haze from Southeast Asian forest fires, with obscured sun and masked city residents.",
"European city under a scorching sun during the 2003 heat wave, streets deserted and hot.",
"Ruins of the Fukushima nuclear plant post-tsunami, under a cloudy sky with radioactive symbols.",
"California forest engulfed in flames at sunset, with firefighters battling the intense wildfire.",
"Stark contrast of lush Amazon rainforest and adjacent deforested barren land with stumps.",
"Polar bear on a melting ice fragment in the Arctic, surrounded by water and distant icebergs.",
"Australian bushfires scene with fleeing kangaroos and a landscape engulfed in red flames.",
"Bleached coral in the Great Barrier Reef, with vibrant living coral and swimming small fish.",
"Sea turtle navigating through ocean cluttered with plastic debris, near a shadowy city skyline.",
"Brazilian Amazon in flames, with rising smoke depicting rainforest destruction.",
"Australian bushfires from above, showing fire consuming forests and causing wildlife distress.",
"California's scorched earth and barren landscapes with wildfires and smoke clouds.",
"East African farmlands overrun by swarms of locusts, devastating crops and causing despair.",
]

prompt_embeds, pooled_prompt_embeds = prompt_encoder(prompts)


time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]] * 2)
aug_embs = torch.stack([aug_embeder(time_ids, text_embeds) for text_embeds in pooled_prompt_embeds])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

prompt_embeds = prompt_embeds.to(device, dtype=dtype)
pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype=dtype)
aug_embs = aug_embs.to(device, dtype=dtype)
pipe_reduced = pipe_reduced.to(device, dtype=dtype)
# pipe_reduced.vae = pipe_reduced.vae.to(device, dtype=torch.float32)

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')
    
    square_ground = Image.open("test/image/background.png")
    mask_img = Image.open("test/image/mask.jpg")
    x, y = 268, 516

    image = (
        torch.tensor(
            np.array(square_ground)[None, ...].transpose((0, 3, 1, 2)),
            device=device,
            dtype=dtype,
        )
        / 255.0
    )
    mask = torch.tensor(np.array(mask_img, dtype=bool).transpose(2, 0, 1)[None, None, 0], device=device, dtype=dtype)

    i = 0
    
    # pipe_reduced = torch.compile(pipe_reduced)
    # pipe_reduced = torch.compile(pipe_reduced, mode="reduce-overhead", fullgraph=True, dynamic=False)
    # print("all reduce-overhead fullgraph static")

    import time
    
    with torch.inference_mode():
        for _ in range(100):
            start_time = time.time()
            img2img_result = pipe_reduced(
                prompt_embeds=prompt_embeds[i],
                aug_emb=aug_embs[i],
                image=image,
                mask=mask,
            )
            torch.cuda.synchronize()
            # torch.cuda.empty_cache()
            print("Time taken: ", time.time() - start_time)

    Image.fromarray(
        (img2img_result[0] * 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    ).save('res.png')
