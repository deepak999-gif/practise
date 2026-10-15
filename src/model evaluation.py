y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Actual values:", y_test.tolist())
print("Predictions:", y_pred.tolist())
print("Baseline accuracy:", accuracy)