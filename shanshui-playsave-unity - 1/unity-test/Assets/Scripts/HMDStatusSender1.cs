using UnityEngine;
using System.Net.Sockets;
using System.Text;

public class HMDStatusSender1 : MonoBehaviour
{
    private Socket clientSocket;
    private PromptController promptController;
    public bool wasUnmounted = true;  // 用于追踪HMD是否处于Unmounted状态
    

    void Start()
    {
        clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        clientSocket.Connect("127.0.0.1", 8080);

        // 获取PromptController的引用
        promptController = FindObjectOfType<PromptController>();

        OVRManager.HMDMounted += OnHMDMounted;
        OVRManager.HMDUnmounted += OnHMDUnmounted;
    }

    void OnHMDMounted()
    {
        if (wasUnmounted)  // 只有在之前是Unmounted状态时，才进行延迟操作
        {
            StartCoroutine(HandleHMDMounted());
        }
    }

    void OnHMDUnmounted()
    {
        if (!wasUnmounted)  // 只有在之前是mounted状态时，才进行延迟操作
        {
            wasUnmounted = true; // 标记为Unmounted状态
            SendMessageToPython("HMD Unmounted");
            if (promptController != null)
            {
                promptController.MuteAudio(); // 静音PromptController
            }
        }
    }

    private System.Collections.IEnumerator HandleHMDMounted()
    {
        yield return new WaitForSeconds(5.0f);  // 等待5秒
        SendMessageToPython("HMD Mounted");

        if (promptController != null)
        {
            promptController.UnmuteAudio();  // 取消静音
            //promptController.PlayNextPrompt();  // 切换到下一个Prompt
        }

        wasUnmounted = false;  // 重置状态
    }

    void SendMessageToPython(string message)
    {
        byte[] data = Encoding.UTF8.GetBytes(message);
        clientSocket.Send(data);
    }

    void OnDestroy()
    {
        clientSocket.Close();
    }
}