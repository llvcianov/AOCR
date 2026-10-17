
import time
import os
import numpy as np
from utils import load_data, write_output, FLOAT_TOLERANCE, SCORE_MULTIPLIER, TQDM_UPDATE_INTERVAL
from algorithms2_copy  import calculate_best_latencies, run_greedy, run_greedy_marginal, run_local_search


# Parametri
INPUT_FILE = r"hashcode_2017_qualification_round.in\me_at_the_zoo.in"
OUTPUT_FILE = r"output\kittens.out"
USE_MARGINAL_GREEDY = True
MAX_PASSES_WITHOUT_IMPROVEMENT = 5  # Stop dopo N pass consecutivi senza miglioramenti
GREEDY_K = 1                     # Max copie per video in tutto il sistema

# Parametri Local Search (per velocità)
MAX_VIDEO_OUT_PER_CACHE = 10   # Max video da provare a rimuovere per cache 
MAX_VIDEO_IN_PER_SWAP = 20     # Max video candidati per swap
MAX_SUCCESSFUL_SWAPS = 5000    # Stop dopo N swap consecutivi andati a buon fine


def main():
    start_time = time.time()
    
    # 1. LOAD DATA
    print(f"--- PHASE 1: LOADING DATA ---")
    data = load_data(INPUT_FILE)
    print(f"Data loaded successfully: {data.V} video, {data.E} endpoint, {data.R} requests, {data.C} caches, cache capacity {data.X}MB")
    print(f"Output file: {OUTPUT_FILE}")
    
    # 2. SEED PER RIPRODUCIBILITÀ
    print(f"\n--- SETTING RANDOM SEED ---")
    np.random.seed(42)
    print("Random seed set to 42 for reproducibility")
    
    # 3. GREEDY
    print(f"\n--- PHASE 2: GREEDY ---")
    if USE_MARGINAL_GREEDY:
        greedy_solution = run_greedy_marginal(data)
    else:
        greedy_solution = run_greedy(data, max_copies_per_video=GREEDY_K)
    
    # 4. STATE AND SCORE
    initial_best_latency, initial_total_saved = calculate_best_latencies(data, greedy_solution)
    initial_score = int((initial_total_saved / data.total_requests) * SCORE_MULTIPLIER)
    print(f"Initial Score (Greedy): {initial_score:,} points")

    # 5. LOCAL SEARCH 
    print(f"\n--- PHASE 3: LOCAL SEARCH ---")
    print(f"Max {MAX_PASSES_WITHOUT_IMPROVEMENT} passes without improvement")
    print(f"Max {MAX_SUCCESSFUL_SWAPS} consecutive successful swaps")
    
    final_solution, final_score = run_local_search(
        greedy_solution, 
        data, 
        max_passes_no_improvement=MAX_PASSES_WITHOUT_IMPROVEMENT,
        best_latency=initial_best_latency,      
        current_total_saved=initial_total_saved,
        max_video_out_per_cache=MAX_VIDEO_OUT_PER_CACHE,
        max_video_in_per_swap=MAX_VIDEO_IN_PER_SWAP,
        max_successful_swaps=MAX_SUCCESSFUL_SWAPS
    )
    
    # 6. RESULTS AND OUTPUT
    print(f"\n--- FINAL RESULT ---")
    print(f"Final Score: {final_score:,} points")
    improvement = final_score - initial_score
    print(f"Local search improvement: +{improvement:,} points")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
    write_output(final_solution, OUTPUT_FILE)
    print(f"\nSolution written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()