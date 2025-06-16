def preprocess_data(df):
    print("[INFO] Starting preprocessing...")

    df = df.dropna()
    df['sepal_ratio'] = df['SepalLengthCm'] / df['SepalWidthCm']
    df['petal_ratio'] = df['PetalLengthCm'] / df['PetalWidthCm']

    X = df.drop(['Species', 'Id'], axis=1)
    y = df['Species']

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"[INFO] Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    return X_train, X_test, y_train, y_test, scaler, label_encoder
