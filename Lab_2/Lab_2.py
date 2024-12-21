import math
import re
from collections import defaultdict, Counter

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


def retrieve_stopwords():
    return set(stopwords.words('english'))


def extract_keywords_from_text(text):
    stop_terms = retrieve_stopwords()
    candidates = re.split(r'[@\"[\-.?!)(,:\]\"]+', text.lower())
    potential_phrases = []
    current_phrase = []

    for candidate in candidates:
        words = re.split(r'\W+', candidate)
        for word in words:
            if word in stop_terms or not word:
                if current_phrase:
                    potential_phrases.append(current_phrase)
                    current_phrase = []
            else:
                current_phrase.append(word)
        if current_phrase:
            potential_phrases.append(current_phrase)
            current_phrase = []

    return potential_phrases


def compute_word_scores(phrases):
    word_count = defaultdict(int)
    word_degree_sum = defaultdict(int)

    for phrase in phrases:
        degree = len(phrase) - 1
        for word in phrase:
            word_count[word] += 1
            word_degree_sum[word] += degree

    for word in word_degree_sum:
        word_degree_sum[word] += word_count[word]

    word_scores = {word: word_degree_sum[word] / word_count[word] for word in word_count}
    return word_scores


def calculate_phrase_scores_for_keywords(phrases, word_scores):
    phrase_scores = {}
    for phrase in phrases:
        phrase_score = sum(word_scores.get(word, 0) for word in phrase)
        phrase_scores[" ".join(phrase)] = phrase_score
    return phrase_scores


def run_rake_algorithm(text):
    phrases = extract_keywords_from_text(text)
    word_scores = compute_word_scores(phrases)
    phrase_scores = calculate_phrase_scores_for_keywords(phrases, word_scores)
    return sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)


def extract_keywords_from_dataset():
    dataset = pd.read_csv('spam.csv', encoding='latin-1')
    dataset = dataset[['v1', 'v2']]
    dataset.columns = ['label', 'message']
    result_data = {'Label': [], 'Keywords': []}

    for _, row in dataset.iterrows():
        ranked_keywords = run_rake_algorithm(row['message'])
        keyword_list = [phrase for phrase, _ in ranked_keywords]
        result_data['Label'].append(row['label'])
        result_data['Keywords'].append(keyword_list)

    result_df = pd.DataFrame(result_data)
    result_df.to_csv('output_keywords.csv', index=False)
    return result_df


def extract_keywords_from_text_input(text):
    ranked_keywords = run_rake_algorithm(text)
    keyword_list = [phrase for phrase, _ in ranked_keywords]
    return keyword_list


def calculate_term_frequency(text):
    tokens = text.split()
    total_tokens = len(tokens)
    frequency = Counter(tokens)
    return {word: count / total_tokens for word, count in frequency.items()}


def calculate_inverse_document_frequency(documents):
    num_docs = len(documents)
    doc_frequency = defaultdict(int)

    for doc in documents:
        unique_terms = set(doc.split())
        for term in unique_terms:
            doc_frequency[term] += 1

    return {term: math.log(num_docs / (1 + freq)) for term, freq in doc_frequency.items()}


def compute_tfidf_scores(corpus):
    idf_values = calculate_inverse_document_frequency(corpus)
    tfidf_results = []

    for doc in corpus:
        tf_values = calculate_term_frequency(doc)
        tfidf = {term: tf_values.get(term, 0) * idf_values.get(term, 0) for term in tf_values}
        tfidf_results.append(tfidf)

    return tfidf_results


def generate_tfidf_for_keywords():
    dataset = extract_keywords_from_dataset()
    dataset['processed_text'] = dataset['Keywords'].apply(lambda x: " ".join(x))
    documents = dataset['processed_text'].tolist()
    tfidf_results = compute_tfidf_scores(documents)
    tfidf_dataframe = pd.DataFrame(tfidf_results).fillna(0)
    tfidf_dataframe['label'] = dataset['Label']
    return tfidf_dataframe


def calculate_single_document_idf(text):
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return {}
    term_frequency = Counter(words)
    idf_values = {word: math.log(1 + word_count / (count + 1)) for word, count in term_frequency.items()}
    return idf_values


def compute_single_document_tfidf(text):
    tf_values = calculate_term_frequency(text)
    idf_values = calculate_single_document_idf(text)
    tfidf_values = {word: tf_values.get(word, 0) * idf_values.get(word, 0) for word in tf_values}
    return tfidf_values


def get_single_document_tfidf(text):
    keywords = extract_keywords_from_text_input(text)
    processed_input = " ".join(keywords)
    tfidf_values = compute_single_document_tfidf(processed_input)
    return tfidf_values


def compute_classification_metrics(actual_labels, predicted_labels):
    true_positive = false_positive = true_negative = false_negative = 0

    for true_label, predicted_label in zip(actual_labels, predicted_labels):
        if true_label == 1 and predicted_label == 1:
            true_positive += 1
        elif true_label == 0 and predicted_label == 1:
            false_positive += 1
        elif true_label == 0 and predicted_label == 0:
            true_negative += 1
        elif true_label == 1 and predicted_label == 0:
            false_negative += 1

    return true_positive, false_positive, true_negative, false_negative


class SpamClassifier(nn.Module):
    def __init__(self, input_size):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


def evaluate_model(model, test_loader, test_data):
    model.eval()
    correct = 0
    total = 0
    predictions_list = []
    actual_labels = []
    keywords_list = []
    correct_predictions = []

    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(test_loader):
            outputs = model(X_batch)
            predictions = (outputs > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
            predictions_list.extend(predictions.cpu().numpy().flatten())
            actual_labels.extend(y_batch.cpu().numpy().flatten())
            keywords_list.extend(
                test_data['message'].iloc[i * test_loader.batch_size:(i + 1) * test_loader.batch_size].values)
            correct_predictions.extend(predictions.cpu().numpy().flatten() == y_batch.cpu().numpy().flatten())

    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    TP, FP, TN, FN = compute_classification_metrics(actual_labels, predictions_list)

    print(f"True Positives: {TP}, False Positives: {FP}, True Negatives: {TN}, False Negatives: {FN}")

    result_df = pd.DataFrame({
        'Keywords': keywords_list,
        'Actual_Label': actual_labels,
        'Predicted_Label': predictions_list,
        'Correct': correct_predictions
    })
    result_df.to_csv('predictions_with_keywords.csv', index=False)
    print("Predictions with keywords saved to 'predictions_with_keywords.csv'")


def vectorize_single_input(single_tf_idf_result, all_features):
    feature_vector = [single_tf_idf_result.get(word, 0) for word in all_features]
    return feature_vector


def evaluate_single_input(model, text, all_features):
    tf_idf_vector = get_single_document_tfidf(text)
    feature_vector = vectorize_single_input(tf_idf_vector, all_features)
    input_tensor = torch.tensor([feature_vector], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output > 0.5).float().item()

    label = "Spam" if prediction == 1.0 else "Not Spam"
    print(f"Text: {text}")
    print(f"Prediction: {label} (Model output: {output.item():.4f})")


def main():
    # nltk.download('stopwords')
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

    tfidf_df = generate_tfidf_for_keywords()
    label_encoder = LabelEncoder()
    tfidf_df['label_encoded'] = label_encoder.fit_transform(tfidf_df['label'])
    X = tfidf_df.drop(columns=['label', 'label_encoded']).values
    y = tfidf_df['label_encoded'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X_train.shape[1]
    model = SpamClassifier(input_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, epochs=10)
    evaluate_model(model, test_loader, data.iloc[X_train.shape[0]:])

    all_features = tfidf_df.drop(columns=['label', 'label_encoded']).columns.tolist()

    example_text = (r"December only! Had your mobile 11mths+? You are entitled to update to the latest colour "
                    r"camera mobile for Free! Call The Mobile Update Co FREE on 08002986906,,,")
    evaluate_single_input(model, example_text, all_features)

    example_text = r"Coffee cake, i guess...,,,"
    evaluate_single_input(model, example_text, all_features)

    example_text = (r"Text & meet someone sexy today. U can find a date or even flirt its up to U. Join 4 just 10p. "
                    r"REPLY with NAME & AGE eg Sam 25. 18 -msg recd@thirtyeight pence,,,")
    evaluate_single_input(model, example_text, all_features)


if __name__ == '__main__':
    main()
