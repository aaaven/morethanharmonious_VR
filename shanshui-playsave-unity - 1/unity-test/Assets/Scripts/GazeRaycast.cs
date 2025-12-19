using UnityEngine;
using System;
using System.Net.Sockets;

public class GazeRaycast : MonoBehaviour
{
    public GameObject resultCanvas;
    public float maxDistance = 1000f;
    public float minCircleRadius = 10f;
    public float maxCircleRadius = 30f;
    public float speedScaleFactor = 0.1f;

    private Camera cam;
    private Vector2 lastUVPosition;
    private bool hasLastUVPosition = false;
    private TcpClient client;
    private NetworkStream stream;
    public string serverIP = "127.0.0.1";
    public int serverPort = 12346;
    private float sendInterval = 0.002f; // 每10毫秒发送一次
    private float timeSinceLastSend = 0f;

    void Start()
    {
        cam = GetComponent<Camera>();
        ConnectToServer();
    }

    void Update()
    {
        Ray ray = new Ray(cam.transform.position, cam.transform.forward);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, maxDistance))
        {
            if (hit.collider.gameObject == resultCanvas)
            {
                Vector2 pixelUV = hit.textureCoord;
                pixelUV.x *= 1047;
                pixelUV.y *= 1544;

                pixelUV.x = 1047 - pixelUV.x;

                if (hasLastUVPosition)
                {
                    float distance = Vector2.Distance(pixelUV, lastUVPosition);
                    float speed = distance / Time.deltaTime;
                    float radius = Mathf.Clamp(maxCircleRadius - speed * speedScaleFactor, minCircleRadius, maxCircleRadius);

                    timeSinceLastSend += Time.deltaTime;
                    if (timeSinceLastSend >= sendInterval)
                    {
                        SendXYCoordinates((int)pixelUV.x, (int)pixelUV.y, speed);
                        timeSinceLastSend = 0f;
                    }
                }

                lastUVPosition = pixelUV;
                hasLastUVPosition = true;
            }
        }
        else
        {
            hasLastUVPosition = false;
        }
    }

    void ConnectToServer()
    {
        try
        {
            client = new TcpClient(serverIP, serverPort);
            stream = client.GetStream();
            Debug.Log("Connected to server.");
        }
        catch (Exception e)
        {
            Debug.LogError("Socket error: " + e.Message);
        }
    }

    void SendXYCoordinates(int x, int y, float speed)
    {
        if (stream != null && stream.CanWrite)
        {
            try
            {
                string message = $"{x},{y},{speed}\n";
                byte[] data = System.Text.Encoding.UTF8.GetBytes(message);
                stream.Write(data, 0, data.Length);
            }
            catch (Exception e)
            {
                Debug.LogError("Error sending data: " + e.Message);
            }
        }
    }

    void OnApplicationQuit()
    {
        if (stream != null) stream.Close();
        if (client != null) client.Close();
    }
}
