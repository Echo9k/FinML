from collections import defaultdict
from typing import List, Dict, Tuple

# Define constants for better readability and to eliminate magic numbers
DEFAULT_THRESHOLD = 0.5
VALID_LABELS = {'ham', 'spam'}


def confusion_matrix(actual_labels, predicted_labels, labels):
    """
    Compute a confusion matrix for a binary classification problem.

    Parameters:
    - actual_labels (list): The true labels of the data.
    - predicted_labels (list): The labels predicted by the classifier.
    - labels (list): The list of unique labels, e.g., ['ham', 'spam'].

    Returns:
    - A dictionary containing the confusion matrix values: {'TP': ..., 'TN': ..., 'FP': ..., 'FN': ...}
    """
    # Initialize counts
    TP = TN = FP = FN = 0

    # Iterate through all the labels and update counts
    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual == labels[1] and predicted == labels[1]:
            TP += 1
        elif actual == labels[0] and predicted == labels[0]:
            TN += 1
        elif actual == labels[0] and predicted == labels[1]:
            FP += 1
        elif actual == labels[1] and predicted == labels[0]:
            FN += 1

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


def _validate_new_data(new_data: Dict[str, List[str]]) -> None:
    """Validates the new data for the update_and_learn method."""
    if not isinstance(new_data, dict):
        raise TypeError("New data should be a dictionary.")
    if any(label.lower() not in VALID_LABELS for label in new_data):
        raise ValueError(f"Invalid label in new data. Valid labels are: {', '.join(VALID_LABELS)}.")


class Classifier:
    """
    A class to perform Naive Bayes algorithm to classify emails as spam or ham based on the word count and probabilities. The classifier can be updated with new data to improve its accuracy.

    Methods:
        __init__(ham_corpus, spam_corpus): Initializes the NaiveBayesClassifier with the given ham and spam data.
        classify(email, threshold): Classifies an email as either "spam" or "ham" based on the calculated probabilities.
        update_and_learn(new_data): Updates the word count and learns from new data to improve the email classifier.

    Examples:
        from NaiveBayes import Classifier
        >>> classifier = Classifier()
        >>> classifier.train(training_data)
        >>> classification = classifier.classify(new_email)
        >>> print(classification)
        "spam"
    """
    def __init__(self, ham_corpus: List[str], spam_corpus: List[str]) -> None:
        """
        Initializes the NaiveBayesClassifier with the given ham and spam data.

        Args:
            ham_corpus (List[str]): A list of ham emails.
            spam_corpus (List[str]): A list of spam emails.

        Returns:
            None
        """
        self._validate_input(ham_corpus, spam_corpus)
        self._ham_word_count = self._count_words(ham_corpus)
        self._spam_word_count = self._count_words(spam_corpus)
        self._update_probabilities()

    @staticmethod
    def _validate_input(ham_corpus: List[str], spam_corpus: List[str]) -> None:
        """Validates the initial data for ham and spam."""
        if not ham_corpus or not spam_corpus:
            raise ValueError("Initial data for ham and spam must be non-empty.")


    def _count_words(self, emails: List[str]):
        """Counts the words in the given list of emails."""
        word_count = defaultdict(int)
        for email in emails:
            word_count = self._update_word_count(email, word_count)
        return word_count

    @staticmethod
    def _update_word_count(email: str, word_count: dict) -> dict:
        """Updates the word count for the given email."""
        for word in email.split():
            word_count[word] += 1
        return word_count

    def _update_probabilities(self) -> None:
        """Updates the probabilities for each word in the word count."""
        self._total_ham_words = sum(self._ham_word_count.values())
        self._total_spam_words = sum(self._spam_word_count.values())
        self._ham_probs = self._calculate_probabilities(self._ham_word_count, self._total_ham_words)
        self._spam_probs = self._calculate_probabilities(self._spam_word_count, self._total_spam_words)

    @staticmethod
    def _calculate_probabilities(word_count: Dict[str, int], total_count: int) -> Dict[str, float]:
        """Calculates the probabilities for each word in the given word count."""
        # Laplace smoothing
        return {word: (count + 1) / (total_count + len(word_count)) for word, count in word_count.items()}

    def classify(self, email: str, threshold: float = DEFAULT_THRESHOLD) -> str:
        """
        Classifies an email as either "spam" or "ham" based on the calculated probabilities.

        Args:
            email (str): The email content to classify.
            threshold (float): The threshold value for classifying as "spam" (default: DEFAULT_THRESHOLD).

        Returns:
            str: The classification label ("spam" or "ham").

        Examples:
            from NaiveBayes import Classifier
            >>> classifier = Classifier()
            >>> classifier.train(training_data)
            >>> classification = classifier.classify(new_email)
            >>> print(classification)
            "spam"
        """

        ham_prob, spam_prob = self._calculate_email_probabilities(email)
        return "spam" if spam_prob / (ham_prob + spam_prob) < threshold else "ham"

    def _calculate_email_probabilities(self, email: str) -> Tuple[float, float]:
        ham_prob = self._product([self._ham_probs.get(word, 1) for word in email.split()])
        spam_prob = self._product([self._spam_probs.get(word, 1) for word in email.split()])
        return ham_prob, spam_prob

    @staticmethod
    def _product(iterable: List[float]) -> float:
        """
        Calculates the product of the given iterable.

        Args:
            iterable (List[float]): The iterable to calculate the product of.

        Returns:
            float: The product of the given iterable.
        """
        result = 1
        for x in iterable:
            result *= x
        return result

    def update_and_learn(self, new_data: Dict[str, List[str]]) -> None:
        """
        Updates the word count and learns from new data to improve the email classifier.

        Args:
            new_data (Dict[str, List[str]]): A dictionary containing the new data to learn from, where the keys are the labels ("ham" or "spam") and the values are lists of emails.

        Returns:
            None

        Note: This method validates the new data, updates the word count based on the emails, and updates the probabilities for classification.

        Examples:
            >>> new_data = {
            ...     "ham": [ham_email_content_1, ham_email_content_2],
            ...     "spam": [spam_email_content_1, spam_email_content_2]
            ... }
            >>> classifier.update_and_learn(new_data)
        """

        _validate_new_data(new_data)
        for label, emails in new_data.items():
            word_count = self._ham_word_count if label.lower() == "ham" else self._spam_word_count
            for email in emails:
                word_count = self._update_word_count(email, word_count)
        self._update_probabilities()
