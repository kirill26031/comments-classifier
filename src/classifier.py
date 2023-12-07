from sklearn.naive_bayes import GaussianNB


def train(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb
