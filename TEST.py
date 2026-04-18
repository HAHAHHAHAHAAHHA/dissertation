import ollama  
import json
import csv
import random
from datetime import datetime
from pathlib import Path

class ResponseRater:                 # prompt file needs to be changed here, bin file here
    def __init__(self, prompts_file="prompts_deduped.csv", used_prompts_file="used_prompts.csv", num_responses=3):
        self.prompts_file = prompts_file
        self.used_prompts_file = used_prompts_file
        self.num_responses = num_responses
 
    def load_prompts(self):        
        prompts = []
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f: 
                reader = csv.reader(f)
                for row in reader:
                    if row:                              # skips empty rows
                        prompts.append(row[0])
        except FileNotFoundError:
            print(f"{self.prompts_file} not found.")                     # just in case i mess up the name
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                pass
        return prompts
    
    def save_prompts(self, prompts):
        with open(self.prompts_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for prompt in prompts:
                writer.writerow([prompt])                 # makes sure to save remaining promtps
    
    def move_to_used(self, prompt):
        with open(self.used_prompts_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([prompt, datetime.now().isoformat()])       # does as advertised,
        
        print(f"✓ Moved prompt to {self.used_prompts_file}")
    
    def query_ollama(self, prompt, system_message="You are a helpful assistant."):  # query to make the model behave as , you guessed it, an assistant
        response = ollama.generate(        # handles backend, thanks ollama
            model="qwen3:0.6b",
            prompt=prompt,
            system=system_message,
            stream=False,
            options={                         # hyperparams
                'temperature': 1.2,
                'top_p':0.8,
                'top_k':20,
                'num_ctx': 8192,
                'repeat_penalty':1.1
                }
        )
        return response['response']
    
    def rate_response(self, prompt, response, response_num, total_responses):
        print("\n" + "="*80)
        print(f"PROMPT: {prompt}")
        print("-"*80)
        print(f"RESPONSE {response_num}/{total_responses}:")
        print(response)
        print("="*80)
        
        ratings = {}
        categories = {
            "brevity": "How concise and exact is the terminology?",
            "safety": "How safe/culturally sensitive is the response?",
            "coherence": "How clear and well-structured is the response?" 
        }
        
        for category, description in categories.items():
            while True:
                try:
                    rating = input(f"\n{description}\nRate {category} (1-10, or 's' to skip this example): ")
                    if rating.lower() == 's':
                        return None   # skipper 
                    rating = int(rating)
                    if 1 <= rating <= 10:
                        ratings[category] = rating
                        break
                    else:
                        print("Please enter a number between 1 and 10")
                except ValueError:
                    print("Invalid input. Enter a number 1-10 or 's' to skip")
        
        # input("\nOptional feedback/notes (press Enter to skip): ").strip() < might need do not remove
        total_score = sum(ratings.values())
        
        return {
            "prompt": prompt,
            "response": response,
            "ratings": ratings,
            "total_score": total_score,
            "average_rating": total_score / len(ratings),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_ranked_responses(self, prompt, rated_responses):
        """Save ranked responses to separate JSON files"""
        if not rated_responses:
            return
        
        ranked = sorted(rated_responses, key=lambda x: x['total_score'], reverse=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for rank, response_data in enumerate(ranked, 1):
            filename = f"ranked_{rank}_score_{response_data['total_score']}_{timestamp}.json"
            
            training_example = {
                "rank": rank,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": response_data["prompt"]},
                    {"role": "assistant", "content": response_data["response"]}
                ],
                "metadata": {
                    "total_score": response_data["total_score"],
                    "ratings": response_data["ratings"],
                    "average_rating": response_data["average_rating"],                    
                    "timestamp": response_data["timestamp"]
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(training_example, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Rank {rank} (Score: {response_data['total_score']}) saved to {filename}")
    
    def process_prompt(self, prompt):
        """Generate multiple responses and rate them"""
        print(f"\n\n{'='*80}")
        print(f"PROCESSING PROMPT:")
        print(f"{prompt}")
        print(f"{'='*80}")
        
        rated_responses = []
        
        for i in range(self.num_responses):
            print(f"\n--- Generating response {i+1}/{self.num_responses} ---")
            
            try:
                response = self.query_ollama(prompt)
                rating_data = self.rate_response(prompt, response, i+1, self.num_responses)
                
                if rating_data is None:
                    print("Skipping all remaining responses for this prompt...")
                    return "skipped"
                
                rated_responses.append(rating_data)
                
            except Exception as e:
                print(f"Error generating {i+1}: {e}")
                continue
        
        if rated_responses:
            print(f"\n--- Saving ranked responses ---")
            self.save_ranked_responses(prompt, rated_responses)
            
            print("\n--- RANKING SUMMARY ---")
            ranked = sorted(rated_responses, key=lambda x: x['total_score'], reverse=True)
            for rank, resp in enumerate(ranked, 1):
                print(f"Rank {rank}: Total Score = {resp['total_score']}, Avg = {resp['average_rating']:.2f}")
            
            return True
        else:
            print("No responses were rated for this prompt.")
            return False
    
    def run(self):
        print(f"Loading prompts from {self.prompts_file} !")
        prompts = self.load_prompts()
        
        if not prompts:
            print(f"No prompts found in {self.prompts_file}!")
            return
        
        print(f"Loaded {len(prompts)} prompts")
        
        while prompts:
            print(f"\n{len(prompts)} prompts remaining")
            prompt = random.choice(prompts)
            success = self.process_prompt(prompt)
            
            if success:  
                prompts.remove(prompt)
                self.move_to_used(prompt)
                self.save_prompts(prompts)
                print(f"prompt removed from active list. {len(prompts)} prompts remaining.")
            
            if not prompts:
                print("who am i kidding this will never happen")
                break
            
            continue_input = input("\n\ngo one more.(y/n): ").lower()
            if continue_input != 'y':
                print(f"THERE ARE prompts remaining in {self.prompts_file}")
                break

if __name__ == "__main__":
    rater = ResponseRater(
        prompts_file="prompts_deduped.csv",
        used_prompts_file="used_prompts.csv",
        num_responses=3
    )
    
    print("starting rating session.")
    rater.run()
    print("\n\ncongrats on rating session")
