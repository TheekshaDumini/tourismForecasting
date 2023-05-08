import handlers
import traceback


def handler(event, context):
    required_keys = ["task", "arguments"]
    request_ok = all([k in required_keys for k in event.keys()])
    if request_ok:
        available_tasks = ["predict", "get_metadata", "get_stats"]
        task = event["task"]
        if task in available_tasks:
            if task == "predict":
                try:
                    return handlers.predict(event["arguments"])
                except Exception as e:
                    print(f"Unidentified exception {e}")
                    traceback.print_exc()
                    return {
                        "status": 500,
                        "headers": {"content-type": "json"},
                        "body": {"message": f"Unknown error occured while prediction"},
                    }
            elif task == "get_metadata":
                return handlers.get_metadata()
            elif task == "get_stats":
                return handlers.get_stats(event["arguments"]["year"])
        else:
            return {
                "status": 400,
                "headers": {"content-type": "json"},
                "body": {
                    "message": f"Invalid task. Available tasks are {available_tasks}"
                },
            }
    else:
        return {
            "status": 400,
            "headers": {"content-type": "json"},
            "body": {
                "message": f"Required keys are not present. Required keys are {required_keys}"
            },
        }
