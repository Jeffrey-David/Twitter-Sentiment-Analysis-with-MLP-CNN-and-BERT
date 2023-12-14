# Twitter-Sentiment-Analysis-with-MLP-CNN-and-BERT

### To use the code download and save the glove_200d to the data folder before trying to run the notebooks.

## 1.	Multi-Layer Perceptron (MLP) 

In Twitter_Sentiment_Analysis_MLP.ipynb, a MLP model is employed for text classification. The MLP is a feedforward neural network consisting of multiple layers, including an input layer, one or more hidden layers, and an output layer. In this case, the input layer receives the TF-IDF vectors representing the text data. The hidden layers perform a series of non-linear transformations on the input data, learning complex patterns and relationships within the features. Finally, the output layer, typically a softmax layer, produces the probability distribution over the target categories. The model is trained to minimise the categorical cross-entropy loss, allowing it to learn to predict the correct category for a given input. MLPs are versatile and can capture both simple and complex relationships in the data, making them a suitable choice for text classification tasks, especially when combined with informative features like TF-IDF vectors.

## 2.	Convolutional Neural Network (CNN)

In 5_3_Twitter_Sentiment_with_CNN-v2.ipynb, a CNN model is used for text classification. CNNs are known for their effectiveness in image processing, but they can also be applied to sequential data, such as text. In this case, the CNN model is designed to take as input the embedding vectors created from pre-trained Glove word embeddings. The model uses convolutional layers to capture local patterns and relationships between words in the text. Max-pooling layers are applied to extract the most relevant features from the convolutional outputs. The final layers, including fully connected layers and a softmax output layer, produce category predictions. CNNs are particularly effective at learning hierarchical features in data, making them well-suited for tasks where local patterns and their composition into higher-level features are crucial, such as text classification.

## 3.	Bidirectional Encoder Representations from Transformers (BERT)

In Sentiment_analysis_with_bert_for_colab_v3.ipynb, a BERT model is used for text classification. It is a transformer-based model that has been pre-trained on massive text corpora and can be fine-tuned for various NLP tasks, including sentiment analysis. The code lays the groundwork for building and training a sentiment analysis model using BERT.

 
## Data Transformation

Data transformation is a critical component of data analysis, especially when dealing with unstructured text data. In this phase, raw textual data is converted into a numerical format that machine learning models can process.

•	Twitter_Sentiment_Analysis_MLP.ipynb focuses on data transformation through feature engineering. Text data is converted into numerical features, known as term frequency-inverse document frequency (TF-IDF) vectors. TF-IDF represents the importance of words in a document relative to the entire corpus, making it a powerful feature representation for text analysis. This transformation allows machine learning models to work with text data effectively. Moreover, the code implements label encoding to convert categorical labels into numerical values, a necessary step in classification tasks.

•	Data transformation in 5_3_Twitter_Sentiment_with_CNN-v2.ipynb is slightly different. It involves loading pre-trained Glove word embeddings, which are dense vectors that represent words in a continuous space. These embeddings capture semantic relationships between words, enabling the model to understand the meaning of words in context. These pre-trained embeddings are used to create an embedding matrix, which maps the words in the dataset to their corresponding word vectors. This not only transforms the text data into numerical format but also enhances the model's ability to capture semantic information within the text.

•	Sentiment_analysis_with_bert_for_colab_v3.ipynb using BERT excels in data transformation for NLP tasks due to its contextual understanding, pretrained representations, and bidirectional capabilities. The code includes tokenization, attention marks, padding, and uses transfer learning to build Sentiment Classifier using the Transformers library by Hugging Face.
