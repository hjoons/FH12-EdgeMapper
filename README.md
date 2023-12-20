# EdgeMapper
## Description
A project that utilizes federated learning to train estimation models on edge devices, in this case an NVIDIA Jetson Nano. The devices are loaded with a pre-trained model and then generate their own local dataset using an Azure Kinect Sensor. The local models are then trained on the local dataset and the updated weights are sent to a central server. The central server then aggregates the weights and sends the updated weights back to the devices. 