from ragas.metrics import faithfulness, answer_relevancy
print(f"Type of faithfulness: {type(faithfulness)}")
print(f"Is instance? {not isinstance(faithfulness, type)}")
print(f"String repr: {faithfulness}")
