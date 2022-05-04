# HAR-on-UCF-Crime-dataset
# Abstract
Surveillance videos can capture a variety of realistic anomalies/unlawfully
activities but there is no real-time analysis of those captured feed. In this project
we propose, two different methods to detect anomaly activities in real-time, 1.
Multiple classes by detecting single activity in real-time and 2. Normal activities
and Unlawful activities.
Our proposed algorithm for video time-scale squeezing was able to standardize the
UCF-Crime data-set by rewriting each video with a fixed time length of 1 second in
total creating a new standardized data-set.The proposed methods were trained on a
3D residual neural network (ResNet 3D 18) with our unique data prepossessing
method along with our algorithm for data-set augmentation which achieves
significant improvement in anomaly detection performance as compared to the
state-of-the-art approaches. Our prediction method itself is 25-30 times faster than
any other method/script/algorithm available on the Internet.It is capable of
analysing long untrim videos and segmenting the unlawfully/anomaly activities in
an efficient way

# Datset Augmentation 
The original data-set was not standardises to use for
deep-learning purpose so in order to make it suitable we propose an algorithm for
video time-scale squeezing was able to standardize the UCF-Crime data-set by
rewriting each video with a fixed time length of 1 second in total creating a new
standardized data-set of 139 MB.
# Video time-scale squeezing algorithm (VTSA)

![VTSSA](https://user-images.githubusercontent.com/67901446/166623629-605fea59-3dea-436b-96f6-a22aab3dd051.png)

The VTSA algorithm was design to fast-forward a video in order to get the maximum
features in less amount of time.The VTSA algorithm works in three phases.
1. First phase: Getting the time
Getting the time of each video is necessary as each videos has different amount
of time,having a dynamic syntax is what is necessary.
2. Second Phase: Calculating the speed up value and Speeding up
After getting the time of each video we calculate the speed up time by
√
time + 6 where we remove the root of time and add with a constant,thereby
fast-forwarding/speeding-up the video by that amount of time.
17
3. Third Phase: Condition and save
After getting the new time line of the video we put a condition in which the
video gets a new speedup value which is represented as √
time + 0.7 and gets
seed-up until the time is equal to one.
Once the condition is satisfied it breaks the loop and saves the video into .mp4
format or once’s the condition reaches a count up-to 30 it breaks the loop and
saves the video into .mp4 format


# Flow/Block Diagram
![block_dig](https://user-images.githubusercontent.com/67901446/166623516-49b9a3aa-09e2-4b98-9baf-37c68285a5b3.png)

