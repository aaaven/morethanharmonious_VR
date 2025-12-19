from gtts import gTTS
from pydub import AudioSegment

# 定义需要转换为语音的文本列表
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
]

# 定义文本转语音的函数
def text_to_speech(text, index, lang='en', tld='com', slow=False):
    """
    Converts text to speech and saves it as an mp3 file, then converts it to a wav file.

    Args:
        text: The text to be converted to speech.
        index: An index to be used in the filename of the generated audio.
        lang: The language code for the speech (default: 'en').
        tld: The top-level domain for the Google Translate service (default: 'com').
        slow: Whether to generate slow speech (default: False).
    """
    # 首先生成 mp3 文件
    tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
    mp3_file = f"tts/{index}.mp3"
    tts.save(mp3_file)

    # 将 mp3 文件转换为 wav 文件
    sound = AudioSegment.from_mp3(mp3_file)
    wav_file = f"tts/{index}.wav"
    sound.export(wav_file, format="wav")

# 遍历 prompts 列表，并将每个文本转换为语音
for index, prompt in enumerate(prompts):
    text_to_speech(prompt, index)
