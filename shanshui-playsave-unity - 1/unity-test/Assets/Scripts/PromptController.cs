using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class PromptController : MonoBehaviour
{
    private TcpClient client;
    private NetworkStream stream;
    private string host = "127.0.0.1";
    private int port = 13000;
    private int lastPromptIdx = -1;
    private int promptIdx;
    private AudioSource audioSource;

    void Start()
    {
        client = new TcpClient(host, port);
        stream = client.GetStream();
        audioSource = GetComponent<AudioSource>();
        promptIdx = UnityEngine.Random.Range(0, 14);
        PlayNextPrompt(); // 开始播放第一个音频
    }

    public void PlayNextPrompt()
    {
        if (!audioSource.mute)
        {
            promptIdx += 1;
            promptIdx = promptIdx % 14;
            string filePath = $"C:/shanshui/python/tts/{promptIdx}.wav";
            StartCoroutine(PlayAudio(filePath, promptIdx));
        }
        else
        {
            StartCoroutine(WaitForUnmute());
        }
    }
    
    private System.Collections.IEnumerator WaitForUnmute()
    {
        // 等待直到音频取消静音
        while (audioSource.mute)
        {
            yield return null; // 每帧等待
        }
        
        // 取消静音后继续播放下一个音频
        PlayNextPrompt();
    }
    
    System.Collections.IEnumerator PlayAudio(string filePath, int promptIdx)
    {
        using (var www = new WWW("file:///" + filePath))
        {
            yield return www;

            if (string.IsNullOrEmpty(www.error))
            {
                audioSource.clip = www.GetAudioClip(false, true, AudioType.WAV);
                audioSource.Play();

                byte[] data = Encoding.UTF8.GetBytes(promptIdx.ToString());
                stream.Write(data, 0, data.Length);

                // 等待音频播放完成
                yield return new WaitForSeconds(audioSource.clip.length);
                
                PlayNextPrompt();
                
            }
            else
            {
                Debug.LogError($"Error loading audio file: {www.error}");
            }
        }
    }

    public void MuteAudio()
    {
        if (audioSource != null && audioSource.isPlaying)
        {
            audioSource.mute = true;
        }
    }
    
    public void UnmuteAudio()
    {
        if (audioSource != null)
        {
            audioSource.mute = false;
        }
    }

    void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
    }
}
