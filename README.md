# Face Aging Model SOC'25 Project

**Notion Resource:** (Given by the mentor)

This is the main page with the  collection of the resources that I followed systematically. All resources are cited from the Notion page.

[https://happy-armchair-815.notion.site/Face-Aging-Model-1f816f25632d80f79f38ede6cd083d38](https://happy-armchair-815.notion.site/Face-Aging-Model-1f816f25632d80f79f38ede6cd083d38)

---

##  My Weekly Learning Log & Sections Completed

### Week 1 / Section 1 (May 21 – May 27): Learning the Essentials

**Python & Interactive Environments**

* Refreshed Python basics: syntax, data structures, and core libraries (NumPy, Pandas, Matplotlib).
* Practiced in Jupyter Notebook and Google Colab, exploring notebook features and setting up GPU runtimes.
* Watched introductory deep learning overviews to understand neural network motivations and architecture.

**Key Resources:**

1. [Python Crash Course – YouTube](https://youtu.be/kqtD5dpn9C8?si=YVC-_cUaq-ehpS38)
2. [W3Schools Python Intro](https://www.w3schools.com/python/python_intro.asp)
3. [Jupyter vs. Colab Tutorial – YouTube](https://youtu.be/mR8cvcwJ1Ko?si=Cq06NypG_UGcrUR-)
4. [Deep Learning Intro – YouTube](https://www.youtube.com/watch?v=o3bWqPdWJ88&pp=ygUVd2hhdCBpcyBkZWVwIGxlYXJuaW5n)

---

### Week 2 / Section 2 (May 28 – June 3): PyTorch and TensorFlow Fundamentals

**Deep Learning Frameworks**

* Explored **PyTorch** dynamic computation graphs, model creation, and training loops.
* Compared with **TensorFlow/Keras**, focusing on static vs. dynamic graph workflows.
* Learnt how to implement simple feedforward networks in both frameworks to classify MNIST digits.

**Key Resources:**

1. [PyTorch Basics Playlist (first 13 videos)](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
2. [TensorFlow Crash Course – YouTube](https://www.youtube.com/watch?v=5Ym-dOS9ssA&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb)
3. [PyTorch Official Docs](https://pytorch.org/)
4. [TensorFlow Guide](https://www.tensorflow.org/guide)

---

### Week 3 / Section 3 (June 4 – June 10): Convolutional Neural Networks & Image Classification

**Building CNNs**

* Understood convolutional layers: filters, strides, padding, and pooling operations.
* Completed CS231n lectures on CNN theory, then built a CNN on MNIST and CIFAR-10 datasets.
* Analyzed impact of depth and hyperparameters on accuracy and overfitting.

**Key Resources:**

1. [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
2. [CNN Intuition – YouTube](https://www.youtube.com/watch?v=YRhxdVk_sIs)
3. [Practical CNN Tutorial – YouTube](https://www.youtube.com/watch?v=pDdP0TFzsoQ&pp=ygUOY25uIGJ5IHB5dG9yY2g%3D)

---

### Week 4 / Section 4 (June 11 – June 17): Image Preprocessing & Vision Transformers

**Data Pipelines & Transformers**

* Learned image loading and preprocessing using OpenCV and PIL; implemented TorchVision transforms.
* Explored popular facial datasets (UTKFace, CelebA, CACD) and prepared sample splits.
* Introduced to Vision Transformers: patch embeddings, multi-head self-attention, and positional encoding.

**Key Resources:**

1. [OpenCV Python Tutorial](https://youtu.be/oXlwWbU8l2o)
2. [Pillow (PIL) Documentation](https://pillow.readthedocs.io/)
3. [TorchVision Transforms – YouTube](https://youtu.be/-oAOgBvXdJY)
4. [Vision Transformer Overview](https://www.youtube.com/watch?v=cEvIFNmmAjc&list=PLH42YHDxfBrKnEd3n6JGdWScU01a6FCyI)

---

### Week 5 / Section 5 (June 18 – June 24): Introduction to Generative Adversarial Networks (GANs)

**GAN Fundamentals**

* Studied the adversarial framework: generator vs. discriminator, adversarial loss, and training dynamics.
* Learned about common challenges like mode collapse and stabilization techniques (e.g., label smoothing, gradient penalty).
* Implemented a basic GAN on MNIST to visualize generated samples over training epochs.

**Key Resources:**

1. [GANs Explained Visually – YouTube](https://youtu.be/TpMIssRdhco)
2. [What are GANs in Machine Learning? – YouTube](https://www.youtube.com/watch?v=X_UUl4HrRFk)
3. [GAN Tutorial Playlist](https://www.youtube.com/watch?v=OXWvrRLzEaU&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va)

---

##  Upcoming Plan

Based on guidelines and help from the mentor, I now plan to  move from learning to building. First, I’ll finalize and organize the facial aging datasets and set up the face detection and alignment pipeline. Next, I will write simple preprocessing scripts for cropping and normalizing images, and split the data for training and validation. By early July, I’ll begin writing the initial model implementation code—a basic conditional GAN—and establish the training workflow. By  mid-July, I’ll integrate feedback from the mentor to improve identity preservation and monitor results. Finally, if I have time left, I’ll prototype additional enhancements like smoothing age transitions and refining the model structure, iterating as I go toward a working face aging application.
