from sklearn.decomposition import IncrementalPCA


def extract_features(features, targets, num=100):
    batch_size = 1000
    ipca = IncrementalPCA(n_components=num, batch_size=batch_size)
    # for i in range(0, features.shape[0], batch_size):
    #     batch_features = features[i:i + batch_size, :].toarray()
    #     batch_targets = targets.iloc[i:i + batch_size].to_list()
    #     ipca.partial_fit(batch_features, batch_targets)
    #     print('' + str(i) + ' of ' + str(features.shape[0]))
    ipca_result = ipca.fit_transform(features)
    return ipca_result, ipca
