# Introduction for *Video Analysis App*
The application is designed to compare the results of the Recurrent Neural Network (RNN) model with ground truth data for analyzing tennis videos.

The *Video Analysis App* functions as the following steps:

1. **Read RNN JSON files and import Ground Truth JSON files:** 
    Seperately, the user can select and upload a folder containing JSON files generated by the RNN model, and a folder containing JSON files with ground truth annotations. The application will read these files and displays the video name, action count, and actions in the separate tables.   

2. **Compare RNN results with Ground Truth:**  
    The user can initiate a comparison between the RNN results and the ground truth data.   

3. **Generate a benchmark report:**  
    After the comparison, the application generates an HTML report that provides a *Video Overview*, *Benchmark Statistics*, and detailed comparison results for each video.    

    __*Video Overview*:__ The section provides an overview of the videos analyzed, including the number and file name of videos in both datasets, videos only in RNN, and videos only in Ground Truth.    

    __*Benchmark Statistics*:__ The section shows the percentage of matched actions with the right type, matched actions with the wrong type, and mismatches between the RNN results and ground truth.    

    __*Visualizing action timelines*:__ In the HTML file, the application also generates a side-by-side timeline visualization of the actions detected by the RNN model and the ground truth annotations. The actions are color-coded based on their type (e.g., Forehand Strike, Backhand Strike, Serve), and markers are used to indicate matches, mismatches, and actions existing in one timeline but not the other.   
       
Overall, this application enables an efficient approach to evaluate the effectiveness of the RNN model. This will facilitate further improvements and refinements.  <br><br><br><br>
<p align="right">Author: Yilin Zhao, Ruohan Jiang<p>