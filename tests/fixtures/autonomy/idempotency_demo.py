results = {}
effects = []

def publish_once(operation_key, text):
    if operation_key not in results:
        effects.append(text)
        results[operation_key] = "receipt-1"
    return results[operation_key]

first = publish_once("candidate-1:publish", "demo only")
second = publish_once("candidate-1:publish", "demo only")
assert first == second
assert effects == ["demo only"]
