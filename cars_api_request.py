import requests
import os
import time

def run_cars_api_request(repeats=3, wait_time=3600):
    API_KEY = "c2a876319341d86d00ef5bf2eed293b7"   # Our API key
    BASE = "https://api.wheel-size.com/v2"         # Base URL of the API
    OUT = "project_data/brands_models.csv"         # Output CSV file name

    # Inner function that performs one batch of requests (up to 100 makes)
    def cars_api_request():
        done = set()  # Keeps track of brands already processed

        # If the output file exists, read it to skip already completed brands
        if os.path.exists(OUT):
            with open(OUT, encoding="utf-8") as f:
                next(f)  # Skip header line
                done = {line.split(",")[0] for line in f}  # Get processed brand names

        # Get the list of all car brands ("makes") from the API
        makes = requests.get(f"{BASE}/makes/", params={"user_key": API_KEY}).json()
        makes = makes["data"] if isinstance(makes, dict) and "data" in makes else makes

        # Open the output CSV in append mode (add new lines to the end)
        with open(OUT, "a", encoding="utf-8") as f:
            # If the file is empty, write the header first
            if os.path.getsize(OUT) == 0:
                f.write("brand,model\n")

            new = 0  # Counter for new brands processed in this run

            # Loop through each brand
            for mk in makes:
                if mk["name"] in done:
                    continue  # Skip if already processed

                # Get all models for this brand from the API
                models = requests.get(
                    f"{BASE}/models/", params={"make": mk["slug"], "user_key": API_KEY}
                ).json()

                # Handle possible JSON structure variations
                models = models["data"] if isinstance(models, dict) and "data" in models else models

                # Write each model to the CSV (brand, model)
                for m in models:
                    f.write(f"{mk['name']},{m['name']}\n")

                new += 1  # Count this brand as processed

                # Stop after processing 100 brands to avoid hitting API limit
                if new >= 100:
                    break

    # Repeat the process up to `repeats` times
    for i in range(repeats):
        cars_api_request()  # Run one batch of requests
        print(f"{i} cycle done")  # Log progress

        # If there are more cycles left, wait before the next one
        if i < repeats - 1:
            print("Sleeping for one hour...")
            time.sleep(wait_time)  # Pause (default 3600 seconds)

    print("All Done")  # Final message when everything is finished
