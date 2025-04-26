# PROFOODAI
 solution for automating food label inspection on production lines using AI.
 
**Project Name:** PROFOODAI 

**Description:**  
This repository contains a complete solution for automating food label inspection on production lines using AI. The system detects printed and non-printed expiration dates with 95% accuracy on edge devices, ensuring compliance and reducing manual inspection errors. Built with Azure Custom Vision, optimized for NVIDIA Jetson Nano, and deployed as a TensorRT engine for real-time performance (120 FPS).  


 

![Demo Preview](Demo2.gif) 



## Demo

[Download and watch the demo video](Demo.mp4)





**Key Features:**  
- **Object Detection:** Identifies printed/non-printed expiration dates using Azure Custom Vision.  
- **Edge Optimization:** Converts models to ONNX → TensorRT → Jetson Engine for low-latency inference.  
- **Real-Time Alerts:** Triggers a buzzer for rejected labels (defects).  
- **Compliance-Ready:** Provides audit-ready data for regulatory standards (FDA, ISO).  

**Getting Started:**  
1. **Azure Custom Vision Training:**  
   - Train a model on labeled images using [Azure Custom Vision Studio](https://customvision.ai/).  
   - Export the model as ONNX.  
2. **TensorRT Conversion:**  
   - Optimize the ONNX model for NVIDIA Jetson Nano using [TensorRT](https://developer.nvidia.com/tensorrt).  
3. **Deployment:**  
   - Deploy the TensorRT engine on Jetson Nano for 120 FPS inference.  

**Links:**  
- **Azure Custom Vision Documentation:** [Quickstart Guide](https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/quickstarts/image-classification)   
- **Azure Custom Vision Studio:** [https://customvision.ai/](https://customvision.ai/)  
- **NVIDIA Jetson Nano Developer Kit:** [https://developer.nvidia.com/embedded/jetson-nano](https://developer.nvidia.com/embedded/jetson-nano)  
- **TensorRT Documentation:** [https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)  

**Demo Deck Presentation Links:**  https://stdntpartners-my.sharepoint.com/:p:/g/personal/shakthi_lakmal_studentambassadors_com/EbCdzMLqIk1OkPulliS8lOIBED1NL1RckC21aQ-ba5fCnA?e=KwK57G
**License:** MIT License.  

**Contact:** SHAKTHI.LAKMAL@studentambassadors.com 
