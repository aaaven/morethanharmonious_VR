using UnityEngine;
using System;
using System.Net.Sockets;
using System.Collections;

public class FrameReceiver : MonoBehaviour
{
    public GameObject resultCanvas;  // 用于展示img2img结果的平面
    public string serverIP = "127.0.0.1";
    public int serverPort = 12345;
    public float frameRate = 30f;
    public float timeoutSeconds = 10f;  // 设置10秒超时
    private RenderTexture flippedTexture;
    private Texture2D texture;
    private Renderer resultCanvasRenderer;
    private TcpClient client;
    private NetworkStream stream;
    private float lastReceivedTime;  // 最后一次接收数据的时间

    private const float scale = 0.5f;
    private const int width = 1047;
    private const int height = 1544;
    private const int bufferSize = width * height * 3;

    void Start()
    {
        texture = new Texture2D(width, height, TextureFormat.RGB24, false);
        flippedTexture = new RenderTexture(width, height, 0);
        resultCanvasRenderer = resultCanvas.GetComponent<Renderer>();
        
        ConnectToServer();
        StartCoroutine(ReceiveAndDisplayFrames());
    }

    void ConnectToServer()
    {
        try
        {
            client = new TcpClient(serverIP, serverPort);
            stream = client.GetStream();
            Debug.Log("Connected to server.");
            lastReceivedTime = Time.time;  // 初始化接收时间
        }
        catch (Exception e)
        {
            Debug.LogError("Socket error: " + e.Message);
        }
    }

    IEnumerator ReceiveAndDisplayFrames()
    {
        byte[] imageData = new byte[bufferSize];

        while (true)
        {
            int totalRead = 0, read = 0;
            while (totalRead < imageData.Length)
            {
                if (stream.DataAvailable)
                {
                    read = stream.Read(imageData, totalRead, imageData.Length - totalRead);
                    totalRead += read;
                    lastReceivedTime = Time.time;  // 更新最后接收数据的时间
                }

                if (Time.time - lastReceivedTime > timeoutSeconds)
                {
                    Debug.Log("Timeout: No data received for 10 seconds, exiting...");
                    ExitPlayMode();
                    yield break;  // 结束协程
                }

                yield return null;  // 等待下一帧继续检查
            }

            Debug.Log($"Frame received, displaying...");

            texture.LoadRawTextureData(imageData);
            texture.Apply();
            
            // 使用Graphics.Blit进行水平翻转
            Graphics.Blit(texture, flippedTexture, new Vector2(-1, 1), new Vector2(1, 1));
            resultCanvasRenderer.material.mainTexture = flippedTexture;

            // 发送确认消息
            byte[] response = System.Text.Encoding.UTF8.GetBytes("OK");
            stream.Write(response, 0, response.Length);

        }
    }

    void ExitPlayMode()
    {
        #if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;  // 如果在Unity编辑器中，停止播放
        #else
        Application.Quit();  // 在运行时环境中退出应用
        #endif
    }

    void OnApplicationQuit()
    {
        if (stream != null) stream.Close();
        if (client != null) client.Close();
    }
}
