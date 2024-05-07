# Deep-Q-Learning-in-Autonomous-Driving
## About the project:
There are used two models, a segmentation model and a model trained by DQL, responsible for choosing the next area to segment.
The chosen architecture for the segmentation model was DeepLabV3 with a ResNet50 backbone, a well-established model known for its effectiveness in segmentation tasks. The segmentation model was binary. For the Deep Q-Network (DQN), the model used in the DQL, I built a customized convolutional neural network (CNN). 
The objective was for the agent to utilize the DQN to select the optimal order for segmenting 16 regions. Following this selection, the segmentation model would be employed to segment each region. The agent's task was to prioritize the most valuable regions.


