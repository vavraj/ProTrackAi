import webview
import subprocess

class API:
    def start_roi(self):
        subprocess.Popen(['python3', 'logic/admin1.py'])
        return {"message": "ROI Definition started"}

    def start_process(self):
        subprocess.Popen(['python3', 'logic/admin2.py'])
        return {"message": "Process Definition started"}

    def start_user_mode(self):
        subprocess.Popen(['python3', 'logic/user.py'])
        return {"message": "User Mode started"}

    def get_stats(self):
        # Placeholder stats; replace with real logic
        return {"total": 10, "correct": 7, "incorrect": 3}

if __name__ == '__main__':
    api = API()
    webview.create_window('Hand Tracking System', 'static/index.html', js_api=api)
    webview.start()
