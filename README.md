<table>
  <tr>
    <td align="center">
      <h2>Tracking Coastal Erosion with AI: Land Cover Classification and Dynamics from Space <a id="top"></a> 
</div></h2>
      <p>This project utilizes SENTINEL-2 imagery and advanced machine learning techniques - K-means clustering and Convolutional Neural Networks (CNNs) - to detect and monitor coastal erosion and land use transformations in the Saint-Trojan coastal zone of Western France between 2015 and 2025.</p>
    </td>
    <td>
      <img src="./images/Sentinel-2_L2A-459879377421259-timelapse.gif" alt="Sentinel-2 Timelapse" width="370"/>
    </td>
  </tr>
</table>










<br>  
<br>  

<details>
<summary>Table of Contents</summary>
  
1. [Project Introduction](#1-project-introduction)
2. [Problem Background](#2-problem-background)
3. [The SENTINEL-2 Satellite](#3-the-sentinel-2-satellite)
4. [Machine Learning Methodologies:](#4-machine-learning-methodologies)
   - [K-Means Clustering](#bullet-k-means-clustering)
   - [Convolutional Neural Network (CNN)](#bullet-convolutional-neural-network-cnn)
5. [Datasets Used](#5-datasets-used)
6. [Usage](#6-usage)
   - [Environmental Cost](#bullet-environmental-cost)
   - [Video Tutorial](#bullet-video-tutorial)
7. [Results](#7-results)
8. [Acknowledgements](#8-acknowledgments)
    - [References](#references)
    - [Contact](#contact)

</details>

---

<br>  


## 1. Project Introduction

This project is the final assignment for the GEOL0069 AI4EO course at UCL, aimed at exploring the application of machine learning techniques in Earth Sciences. The focus of this project is on utilizing unsupervised and supervised learning to identify coastal erosion patterns through satellite imagery. SENTINEL-2 data is employed for its high spatial resolution and relevance in coastal monitoring. The primary algorithms used for classification in this project are K-means clustering (for unsupervised classification) and the Convolutional Neural Network (CNN) method (for feature extraction and land type identification), which are applied to analyze and monitor land cover changes over time.




<br>  

## 2. Problem Background

Coastal erosion is a critical environmental issue, threatening ecosystems, human livelihoods, and infrastructure. Traditional monitoring methods, such as field surveys, are costly and time-intensive, making large-scale and real-time analysis difficult. Satellite remote sensing, particularly through high-resolution imagery like SENTINEL-2, provides a solution by offering frequent, accessible, and detailed data on coastal regions (Phiri et al., 2020).




WHY FRANCE (Musereau et al., 2007)



This project focuses on using machine learning algorithms to analyze SENTINEL-2 satellite data for monitoring coastal erosion in the Saint-Trojan Coastal Area in Western France. By applying K-means clustering and Convolutional Neural Networks (CNN), we aim to classify land cover types and detect changes related to erosion over time. These machine learning techniques allow for automated and scalable analysis, providing insights into coastal dynamics and land use changes.

The goal is to enhance coastal management by providing more accurate, data-driven tools for monitoring erosion and supporting decision-making. Ultimately, this project demonstrates the potential of AI in Earth observation, offering an innovative approach to studying coastal erosion and environmental change.




<br>  

## 3. The SENTINEL-2 Satellite

FIGURE ON S2

<br>  


## 4. Machine Learning Methodologies:

### <a name="bullet-k-means-clustering"></a>• K-Means Clustering

FIGURE ON K-MEANS

<br>  


### <a name="bullet-convolutional-neural-network-cnn"></a>• Convolutional Neural Network (CNN)

FIGURE ON CNN

<br>  


## 5. Datasets Used



<br>  

## 6. Usage

```python
pip install rasterio
```

### <a name="bullet-environmental-cost"></a>• Environmental Cost





### <a name="bullet-video-tutorial"></a>• Video Tutorial

Click below to explore the video demonstration, providing an overview of the code's functionality and operation.

[<img src="./images/video.png" alt="Click here to watch the video demonstration" width="400"/>](https://youtu.be/rqpMsphdrzo)





<br>  

## 7. Results




<br>  

## 8. Acknowledgments

This project was developed for GEOL0069 (Artificial Intelligence For Earth Observation) 2024/2025 at UCL, led by the module team: Dr Michel Tsamados, Weibin Chen, and Connor Nelson.

<br>  


## References
*Copernicus Browser.* (n.d.). (Accessed 2025), from the Copernicus Browser website. https://browser.dataspace.copernicus.eu

Molnar C. (2025). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.* https://christophm.github.io/interpretable-ml-book/cnn-features.html

Musereau, J., Regnauld, H., & Planchon, O. (2007). *Vulnerability of coastal dunes to storms: development of a damage prediction model using the example of Saint-Trojan (Île d'Oléron, France).* Climatologie, 4, 145-166. https://climatology.edpsciences.org/articles/climat/full_html/2007/01/climat20074p145/climat20074p145.html

Phiri, D., Simwanda, M., Salekin, S., Nyirenda, V. R., Murayama, Y., & Ranagalage, M. (2020). *Sentinel-2 data for land cover/use mapping: A review.* Remote sensing, 12(14), 2291. https://www.mdpi.com/2072-4292/12/14/2291

Tsamados M. & Chen W. (2022). *Regression Techniques for Predictive Analysis.* GEOL0069 GitHub Page. (Accessed 2025). https://cpomucl.github.io/GEOL0069-AI4EO/Chapter1_Regression.html

Tsamados M. & Chen W. (2022). *Unsupervised Learning.* GEOL0069 GitHub Page. (Accessed 2025). https://cpomucl.github.io/GEOL0069-AI4EO/Chapter1%3AUnsupervised_Learning_Methods.html





<br>  

## Contact

Project Author: Monika Gonciarz monika.gonciarz.22@ucl.ac.uk

Project Link: https://github.com/monikagonciarz/AI4EO_final_project

<br>
<br>



<div style="text-align: right;">
  
  [(Back to top)](#top)
</div>



