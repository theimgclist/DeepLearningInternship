Given enough data and computation, the neural networks can approxiate any given function. With this ability, they have found their way into computer vision tasks and have replaced the existing methods. Computer vision includes a variety of tasks like object detection, object recongnition, facial recognition, moving object detection etc. The process of detecting a moving object is different from that of detecting an object that is static in terms of motion. Just like object detection is a pre-step for object recognition,background subtraction is the common approach for moving objects segmentation and there are several methods to do this.  
In the paper being discussed, they have used a triplet convolutional neural network for foreground segmentation. The moving objec sequence is taken one frame at a time and to detect the object in the foreground, the background must be subtracted. Each frame is converted into images of 3 different scales for feature encoding and are passed through 3 different convolutional neural nets. Each CNN extracts feature maps of each of the scaled frame. All this happens at the encoding side. From here, the extracted f eature maps are concatenated and form the input for decoding, where TCNN(Transposed CNN) is used for decoding. During decoding,TCNN learns the weights for decoding the feature maps and output the dense probability mask which gives a score that helps in classifying each pixel as that of a foreground or a background.
<p><img src="https://github.com/theimgclist/DeepLearningInternship/blob/master/Project/images/triplet.png"/></p>
