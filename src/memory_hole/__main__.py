import multiprocessing

import uvicorn

if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run("memory_hole.api:app", host="127.0.0.1", port=8765, log_level="info")
