import re
import multiprocessing


def execute_with_timeout(function, response, result_queue, error_info_queue):
    local_vars = {"response": response}
    try:
        # Step 1: Extract all import statements from the function code
        # import_statements = re.findall(r'^\s*import\s+\S+|\s*from\s+\S+\s+import\s+\S+', function, re.MULTILINE)
        import_statements = re.findall(
            r"^\s*(import\s+\S+|from\s+\S+\s+import\s+\S+)", function, re.MULTILINE
        )

        # Step 2: Execute each import statement to ensure they are available in globals
        for statement in import_statements:
            exec(statement, globals())
            
        # Execute the function in the local_vars scope
        exec(function, globals(), local_vars)

        # Update global variables with the local_vars
        globals().update(local_vars)

        if 'check_following' in local_vars:
            check_following = local_vars['check_following']
            if callable(check_following):
                result = check_following(local_vars['response'])
                result_queue.put(result)  
            else:
                raise ValueError("check_following is not a callable function.")
        else:
            result_queue.put(None)
    except Exception as e:
        error_message = f"Error executing function {function}: {e}"
        print(error_message)
        error_info_queue.put(error_message)  
        result_queue.put(None)  


def execute_code(response, function=None):
    result, error_info = None, None

    # Use multiprocessing to handle timeout
    result_queue = multiprocessing.Queue()
    error_info_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=execute_with_timeout, 
        args=(function, response, result_queue, error_info_queue)
    )
    
    process.start()
    process.join(timeout=5)

    if process.is_alive():
        timeout_message = f"Execution timed out for function {function}"
        print(timeout_message)
        process.terminate()  
        process.join()
        error_info = timeout_message
    else:
        result = result_queue.get() if not result_queue.empty() else None
        error_info = error_info_queue.get() if not error_info_queue.empty() else None

    return result, error_info

