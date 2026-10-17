from collections import defaultdict
from typing import Callable, Dict, Optional, Set, Tuple, List
import heapq
from utils import (ProblemData, print_stats, update_latencies_for_video_in_cache,
                   is_swap_valid, find_second_best_latency, get_stochastic_candidates,
                   FLOAT_TOLERANCE, SCORE_MULTIPLIER, TQDM_UPDATE_INTERVAL)
from tqdm import tqdm
import numpy as np


def compute_benefit(video_id: int, cache_id: int, data: ProblemData, 
                    current_best_latency: Optional [Dict[Tuple[int, int], int]] = None) -> Tuple[float, float]:
    """
    Calcola il beneficio statico o marginale (a seconda del parametro current_best_latency) 
    di aggiungere un video a una cache.
    
    INPUT:
        video_id: ID del video da valutare.
        cache_id: ID della cache in cui considerare il piazzamento.
        data: Struttura dati contenente le informazioni del problema.
        (Opzionale) current_best_latency: Dizionario (video_id, endpoint_id) -> miglior latenza attuale.
    
    OUTPUT:
        beneficio totale: Tempo totale risparmiato in millisecondi.
        beneficio statico per video: Tempo risparmiato normalizzato per dimensione video.
    
    COMPORTAMENTO:
        - Per ogni endpoint connesso alla cache in input:
            - Verifica se ci sono richieste per il video.
            - Se la funzione è chiamata da run greedy
                - Calcola il risparmio di latenza rispetto al datacenter
            - Se la funzione è chiamata da run greedy marginale
                - Calcola il risparmio rispetto alla miglior latenza corrente (già ottenuta da altre cache o dal DC)
            - Somma il beneficio pesato per numero di richieste.
        - Normalizza dividendo per la dimensione del video.
    """
    video_size = data.video_sizes[video_id]
    if video_size == 0: 
        return 0.0, 0.0
    
    total_benefit = 0
    
    # Iteriamo solo sugli endpoint connessi alla cache in input
    for endpoint_id in data.cache_to_endpoints[cache_id]:
        
        # 1. Controlliamo se questo endpoint vuole il video
        num_requests = data.requests_map.get((video_id, endpoint_id), 0)
        if num_requests == 0: 
            continue
        
        # 2. Recuperiamo le latenze della cache e del dc di questo endpoint
        cache_lat = data.cache_endpoint_latency[(cache_id, endpoint_id)]
        dc_latency = data.endpoints[endpoint_id]['datacenter_latency']

        # 3. Calcolo beneficio
        # In base al parametro confrontiamo rispetto a dc oppure rispetto alla best attuale
        if current_best_latency is not None:
            compare_lat = current_best_latency.get((video_id, endpoint_id), dc_latency)
        else:
            compare_lat = dc_latency
       
        if cache_lat < compare_lat:
            latency_saved = compare_lat - cache_lat
            total_benefit += num_requests * latency_saved
            
    return total_benefit, (total_benefit / video_size)


def build_initial_heap(data: ProblemData) -> List:
    """
    Costruisce la heap iniziale calcolando il beneficio statico (rispetto al Data Center).
    
    INPUT:
        data: Oggetto ProblemData con le strutture dati.
        
    OUTPUT:
        heap: Lista usata come Priority Queue con tuple (-ratio, video_id, cache_id).

    PSEUDOCODICE:
        - Per ogni Cache C:
            - Identifica i video V richiesti dagli endpoint connessi
                - Se V entra fisicamente in C:
                    - Calcola Ratio = Beneficio(V, C) / Dimensione(V)
                    - Check Ratio > 0 
                    - Aggiungi (-Ratio, V, C) alla Heap

        3. Return: Restituisce la heap popolata e la sua dimensione.
    """
    heap = []

    # 1. Totale candidati
    total_candidates = sum(len(data.cache_to_requested_videos[c]) for c in range(data.C))

    # 2. Avviamo la barra
    # miniters aggiorna la grafica ogni tanto per non rallentare troppo
    with tqdm(total=total_candidates, desc="Building Heap", unit="cand", miniters=TQDM_UPDATE_INTERVAL) as pbar:
    
        # 3. cicla su tutte le cache e i video candidati
        for cache_id in range(data.C):
            # Prendiamo solo i video richiesti dagli endpoint di questa cache
            candidates = data.cache_to_requested_videos[cache_id]
            
            for video_id in candidates:
                # Check fisico di capacità
                if data.video_sizes[video_id] <= data.X:
                    # Calcolo beneficio statico (confronto con Data Center)
                    tot_ben, ratio = compute_benefit(video_id, cache_id, data)
                    
                    if ratio > 0:
                        # Heap di massimo (usiamo ratio negativo, di default è min-heap)
                        heapq.heappush(heap, (-ratio, video_id, cache_id, tot_ben))                        
            
            pbar.update(len(candidates))

    print(f"Heap populated with {len(heap):,} candidates.")                                
    return heap


def run_greedy(data: ProblemData, max_copies_per_video: int) -> Dict[int, Set[int]]:
    """
    Esegue l'algoritmo Greedy basato sul beneficio assoluto.
    
    INPUT:
        data: Oggetto ProblemData contenente tutte le strutture dati del problema 
        max_copies_per_video (int): Limite euristico 'K' per il numero massimo di copie 
                                    dello stesso video ammesse in tutta la rete.

    OUTPUT:
        cache_assignments: Dizionario cache_id -> set di video_id assegnati.

    PSEUDOCODICE:
        1. Inizializza strutture:
            - cache_assignments (vuoto)
            - heap (coda di priorità vuota)
            - contatori copie video
        
        2. Popolamento Heap

        3. Loop Principale:
            - Finché la Heap non è vuota:
                - Estrai candidato migliore (V, C) con ratio massimo
                - Verifica Vincoli:
                    a. Spazio residuo in C sufficiente?
                    b. V non è già presente in C?
                    c. Copie totali di V < max_copies_per_video?
                - Se tutti i vincoli sono soddisfatti:
                    - Assegna V a C
                    - Aggiorna spazio occupato e contatore copie
        
        4. Ritorna cache_assignments
    """
    print(f"Starting Greedy (K={max_copies_per_video})")
    
    # 1. Setup Strutture
    cache_assignments = defaultdict(set)
    cache_used_space = np.zeros(data.C, dtype=int)
    video_copy_count = defaultdict(int) # video_id -> numero di copie assegnate

    # 2. Riempimento Heap
    heap = build_initial_heap(data)

    # Contatori
    metric_pop = 0

    # 4. Ciclo Greedy 
    with tqdm(total=len(heap), desc="Greedy Loop Exec", unit="step") as pbar:
        while heap:
            #ignoriamo il beneficio totale (_) nel greedy statico
            _, video_id, cache_id, _ = heapq.heappop(heap)
            metric_pop += 1
            pbar.update(1)

            # A. Vincolo Spazio
            if cache_used_space[cache_id] + data.video_sizes[video_id] > data.X:
                continue
                
            # B. Vincolo Duplicati Locali (è già qui?)
            if video_id in cache_assignments[cache_id]:
                continue
                
            # C. Vincolo Copie Globali (K) 
            if video_copy_count[video_id] >= max_copies_per_video:
                continue
                
            # Assegnazione a questa cache
            cache_assignments[cache_id].add(video_id)
            cache_used_space[cache_id] += data.video_sizes[video_id]
            video_copy_count[video_id] += 1
    
    # 5. Report
    print("Greedy Done.")
    print(f"[Heap stats] Total Pops: {metric_pop:,}")
    print_stats(cache_assignments, cache_used_space, data)
    
    return cache_assignments


def run_greedy_marginal(data: ProblemData) -> Dict[int, Set[int]]:
    """
    Esegue l'algoritmo Greedy basato sul beneficio marginale.
    Il beneficio marginale risolve la seguente problematica:
    Supponiamo di avere due cache C1 e C2 connesse allo stesso endpoint E.
    L'heap viene popolata con i benefici calcolati rispetto al Data Center.
    Se durante il while assegniamo un video V a C1, il beneficio di assegnare V a C2 diminuisce,
    poiché le richieste per V da E ora hanno già una latenza migliore (C1).
    Quindi la coppia (V, C2) nella heap ha un beneficio obsoleto.
    Per risolvere questo problema, ad ogni estrazione dalla heap ricalcoliamo il beneficio.

    INPUT:
        data: Oggetto ProblemData contenente tutte le strutture dati del problema

    OUTPUT:
        cache_assignments: Dizionario cache_id -> set di video_id assegnati.

    PSEUDOCODICE:
        1. Inizializza strutture:
            - cache_assignments (vuoto)
            - heap (coda di priorità vuota)
            - current_best_latency (mappa video-endpoint -> miglior latenza attuale)
        
        2. Popolamento Heap

        3. Loop Principale:
            - Finché la Heap non è vuota:
                - Estrai candidato migliore (V, C) con ratio massimo
                    - A. Se V è già presente in C: scarta
                - Ricalcola beneficio marginale reale con current_best_latency
                    - B.Se beneficio <= 0: scarta
                    - C.Se il beneficio è cambiato significativamente: reinserisci nella heap col nuovo valore
                      (non ha senso scartarlo, potrebbe essere ancora valido più tardi, non buttiamo via punti!))
                    - D.Se il Vincolo Spaziale non è rispettato: scarta
                - Se ok:
                    - Assegna V a C
                    - Aggiorna spazio occupato
                    - Aggiorna current_best_latency per gli endpoint interessati
    
        4. Ritorna cache_assignments

    """
    print("Starting Marginal Greedy")
    
    # 1. Setup Strutture
    cache_assignments = defaultdict(set)
    cache_used_space = np.zeros(data.C, dtype=int)
    # Inizializzazione dizionario (video_id, endpoint_id) -> miglior latenza attuale
    # All'inizio la best latency per tutti è il Data Center
    current_best_latency = {
        (v, e): data.endpoints[e]['datacenter_latency'] for (v, e) in data.requests_map.keys()
    }

    # 2. Riempimento Heap
    # Anche se abbiamo current_best_latency, all'inizio è uguale al Data Center,
    # quindi il beneficio calcolato è identico al greedy statico
    heap = build_initial_heap(data)

    # Contatori
    metric_reinsertions = 0
    metric_pop = 0

    # 4. Ciclo Greedy Marginal
    with tqdm(total=len(heap), desc="Marginal Greedy Loop Exec", unit="step") as pbar:
        while heap:

            #Estrae il candidato con il beneficio stimato più alto
            neg_ratio, video_id, cache_id, _ = heapq.heappop(heap)
            metric_pop += 1
            pbar.update(1)

            # A. Vincolo Duplicati Locali (è già qui?)
            if video_id in cache_assignments[cache_id]:
                continue
            
            # Ricalcolo beneficio con lo stato attuale delle latenze migliori
            # ho estratto il candidato migliore, ma il beneficio potrebbe essere cambiato dopo alcune assegnazioni
            real_ben, real_ratio = compute_benefit(video_id, cache_id, data, current_best_latency)
            
            # B. Se il beneficio è sceso a zero, scartiamo
            if real_ratio <= 0: 
                continue
            
            # C. Se il beneficio è cambiato significativamente rispetto alla stima nella heap
            #    non scartiamo l'opzione, ma rimettiamo nella heap col nuovo valore corretto
            #    (Usiamo una tolleranza relativa per i float)
            if abs(real_ratio - (-neg_ratio)) > FLOAT_TOLERANCE:
                heapq.heappush(heap, (-real_ratio, video_id, cache_id, real_ben))
                metric_reinsertions += 1
                pbar.total += 1
                pbar.refresh()
                continue
            
            # D. Vincolo Spazio
            # Questo controllo deve essere fatto dopo il ricalcolo del beneficio reale
            # la cache potrebbe essersi riempita mentre l'elemento era nella heap
            if cache_used_space[cache_id] + data.video_sizes[video_id] > data.X:
                continue
                
            # Assegnazione a questa cache
            cache_assignments[cache_id].add(video_id)
            cache_used_space[cache_id] += data.video_sizes[video_id]
            
            # Aggiorniamo la mappa delle migliori latenze per gli endpoint connessi a questa cache
            update_latencies_for_video_in_cache(video_id, cache_id, data, current_best_latency)


    # 5. Report finale
    print("Greedy Marginal Done.")
    print(f"[Heap stats] Total Pops: {metric_pop:,} | Total Re-insertions: {metric_reinsertions:,}")
    print_stats(cache_assignments, cache_used_space, data)
    return cache_assignments


def calculate_best_latencies(data: ProblemData, assignments: Dict[int, Set[int]]) -> Tuple[Dict, float]:
    """
    Inizializza la mappa delle migliori latenze attuali e calcola il risparmio totale iniziale.

    INPUT:
        data: Oggetto ProblemData con le strutture del problema.
        assignments: Dizionario cache_id -> set di video assegnati (soluzione greedy).

    OUTPUT:
        best_latency: Dizionario (video_id, endpoint_id) -> latenza minima attuale.
        total_saved: Float rappresentante la somma totale dei millisecondi risparmiati
                     rispetto al Data Center con la configurazione attuale.

    PSEUDOCODICE:
        1. Inizializzazione Base:
           - Per ogni richiesta (Video, Endpoint) nel sistema:
             - Imposta best_latency[(V, E)] = Latenza Data Center (caso peggiore).

        2. Calcolo Miglioramenti Cache:
           - Per ogni Cache C 
                - Calcola il risparmio ottenuto assegnando i video in C

        3. Return best_latency e total_saved.
    """
    best_latency = {}
    
    # 1. Base: Datacenter
    for (v, e) in data.requests_map.keys():
        best_latency[(v, e)] = data.endpoints[e]['datacenter_latency']
        
    # 2. Miglioramento con cache
    total_saved = 0.0
    for c_id, videos in assignments.items():
        for v_id in videos:
            total_saved += update_latencies_for_video_in_cache(v_id, c_id, data, best_latency)

    return best_latency, total_saved


def calculate_swap_delta(video_out: int, video_in: int, cache_id: int, 
                         data: ProblemData, best_latency: Dict, assignments: Dict) -> Tuple[float, Dict[int, int]]:
    """
    Simula uno scambio (Swap) e calcola la variazione netta di score (Delta),
    senza modificare le strutture dati persistenti.

    INPUT:
        video_out: ID del video candidato alla rimozione.
        video_in: ID del video candidato all'inserimento.
        cache_id: ID della cache su cui effettuare lo scambio 
        data: Oggetto ProblemData.
        best_latency: Dizionario (video_id, endpoint_id) -> latenza minima attuale.
        assignments: Soluzione iniziale (dizionario cache -> set di video_id assegnati).

    OUTPUT:
        delta: Float che rappresenta la variazione di tempo risparmiato.
               - Valore > 0: Lo scambio migliora la soluzione.
               - Valore <= 0: Lo scambio è peggiorativo o neutro.
        second_best_map: Dizionario endpoint_id -> nuova miglior latenza per video_out.
                        Serve per evitare di ricalcolare in update_state_after_swap.

    PSEUDOCODICE:
        1. Inizializza gain e loss

        2. Calcolo PERDITA (Rimuovendo video_out):
           - Per ogni Endpoint E connesso alla cache_id:
             - Se la cache corrente forniva la latenza migliore per (video_out, E):
               - Trova la "Second Best" latenza:
                 - Inizia ipotizzando il Data Center.
                 - Controlla tutte le altre cache connesse a E.
                 - Se un'altra cache ha video_out ed è più veloce del DC, aggiorna la new_best.
               - Sottrai dal delta la perdita: (best_attuale - new_best) * richieste.
                 (Nota: best_attuale < new_best, quindi è un valore negativo).
               - Memorizza new_best in second_best_map per riusarlo dopo.

        3. Return delta e second_best_map.
    """
    # Inizializza gain, il risparmio ottenuto aggiungendo video_in
    gain, _ = compute_benefit(video_in, cache_id, data, current_best_latency=best_latency)
    if gain <= 0:
        return -1.0, {}

    
    # Calcolo perdita (Rimuovendo video_out)
    loss = 0
    second_best_map = {}  # endpoint_id -> nuova best latency per video_out
    
    for e in data.cache_to_endpoints[cache_id]:

        # Se non c'è richiesta per video_out da tutti gli endpoint connessi, non c'è perdita
        if (video_out, e) not in data.requests_map:
            continue
        
        # Se questa cache non era la migliore, rimuovere il video_out non fa danni 
        current_lat = data.cache_endpoint_latency[(cache_id, e)]
        if current_lat != best_latency[(video_out, e)]:
            continue

        # Se era la migliore, dobbiamo trovare chi prende il suo posto
        new_best = find_second_best_latency(video_out, e, cache_id, data, assignments)
        second_best_map[e] = new_best  # Salviamo per riusarlo dopo
            
        loss += (new_best - best_latency[(video_out, e)]) * data.requests_map[(video_out, e)]
    
    delta = gain - loss
    return delta, second_best_map


def update_state_after_swap(video_out: int, video_in: int, cache_id: int, 
                            data: ProblemData, best_latency: Dict, assignments: Dict,
                            second_best_map: Dict[int, int]):
    """
    Applica fisicamente lo scambio e aggiorna la mappa best_latency per renderla
    coerente con la nuova configurazione.

    INPUT:
        video_out: ID del video da rimuovere.
        video_in: ID del video da aggiungere.
        cache_id: ID della cache modificata.
        data: Oggetto ProblemData.
        best_latency: Mappa da aggiornare (Modificata in-place).
        assignments: Dizionario assegnamenti da aggiornare (Modificato in-place).
        second_best_map: Dizionario endpoint_id -> nuova best latency per video_out,
                        precalcolato da calculate_swap_delta per evitare ricalcoli.

    OUTPUT:
        Nessuno (Le strutture dati sono modificate per riferimento).

    PSEUDOCODICE:
        1. Aggiorna Assegnamenti:
           - Rimuovi video_out dal set della cache_id.
           - Aggiungi video_in al set della cache_id.

        2. Aggiorna Best Latency per video_in (Miglioramento):
            - Chiama update_latencies_for_video_in_cache per propagare il miglioramento agli endpoint.

        3. Aggiorna Best Latency per video_out (Peggioramento):
           - Per ogni Endpoint E in second_best_map:
             - Usa il valore precalcolato invece di ricalcolare.
    """
    # 1. Modifica Assegnamento
    assignments[cache_id].remove(video_out)
    assignments[cache_id].add(video_in)
    
    # 2. Aggiorna Best Latency per video_in 
    update_latencies_for_video_in_cache(video_in, cache_id, data, best_latency)

    # 3. Aggiorna Best Latency per video_out usando i valori precalcolati
    for e, new_best in second_best_map.items():
        key = (video_out, e)
        best_latency[key] = new_best


def run_local_search(assignments: Dict[int, Set[int]],
                     data: ProblemData,
                     max_passes_no_improvement: int,
                     best_latency: Dict,        
                     current_total_saved: float,
                     max_video_out_per_cache: int = 10,
                     max_video_in_per_swap: int = 20,
                     max_successful_swaps: int = 5000
                     ) -> tuple:
    """
    Esegue la Local Search: tenta di migliorare la soluzione corrente
    mediante swap di video all'interno delle cache
    Il miglioramento è calcolato utilizzando i delta score anzichè ricalcolando lo score completo.

    INPUT:
        assignments: Soluzione iniziale (dizionario cache -> set di video_id assegnati).
        data: Oggetto ProblemData.
        max_iter: Numero massimo di iterazioni senza miglioramenti globali o totali.
        best_latency: Dizionario (video_id, endpoint_id) -> latenza minima attuale.
        current_total_saved: Risparmio totale iniziale in millisecondi.
        max_video_out_per_cache: Max numero di video da provare a rimuovere per cache (default 10).
        max_video_in_per_swap: Max numero di video candidati da provare per ogni rimozione (default 20).
        max_successful_swaps: Max numero di swap consecutivi andati a buon fine prima di fermarsi (default 5000).

    OUTPUT:
        assignments: La soluzione migliorata (dizionario cache -> set di video_id assegnati)
        current_score: Il punteggio finale raggiunto.

    PSEUDOCODICE:
        - While (ci sono miglioramenti AND iterazioni < max_iter):
            - Randomizza l'ordine delle Cache.
            - Per ogni Cache C nell'ordine:
                - Per ogni video candidato all'uscita (video_out) presente in C:  
                    - Seleziona candidati all'entrata (Candidates IN) dai video richiesti.
                    - Randomizza l'ordine dei candidati IN.
                    
                    - Per ogni video_in nei candidati:
                        - Verifica Vincolo Capacità (Space_Used - Size(Out) + Size(In) <= X).
                        - Chiama calculate_swap_delta(video_out, video_in, ...).
                        
                        - Se Delta > 0 (Miglioramento trovato):
                            - Chiama update_state_after_swap(...).
                            - Aggiorna total_saved_accumulated += Delta.
                            - Segnala "improved = True".
                            - Break (Esce dal ciclo video_in).
                    
                    - Se swap effettuato: Break (Esce dal ciclo video_out e passa alla prossima Cache).

        - Return assignments, current_score.
    """
    print("Starting Local Search (Refactored)...")
    
    # 1. Init
    total_saved_acc = current_total_saved
    current_score = int((total_saved_acc / data.total_requests) * SCORE_MULTIPLIER)   

    #counters
    total_swaps = 0
    consecutive_successful_swaps = 0  # Contatore swap consecutivi andati a buon fine
    pass_count = 0 # Iterazione completa su tutte le cache
    passes_without_improvement = 0
    last_swap_info = None  # Memorizza info ultimo swap per stamparlo a fine pass

    while passes_without_improvement < max_passes_no_improvement and consecutive_successful_swaps < max_successful_swaps:
        pass_count += 1
        swap_found_this_pass = False

        # Randomizzo l'ordine delle cache per non favorire sempre le stesse
        cache_order = np.random.permutation(data.C)

        iterator = tqdm(cache_order, 
                        desc=f"Pass {pass_count}", 
                        leave=False, # leave=False fa scomparire la barra quando finisce/resetta,
                        unit="cache")

        #iteriamo su tutte le cache in ordine random
        for cache_id in iterator:
            if swap_found_this_pass:  # Se abbiamo già trovato uno swap, esci
                break

            # Selezione stocastica di candidati OUT e IN
            videos_in_cache, candidates, used_space = get_stochastic_candidates(
                cache_id, assignments, data, max_video_out_per_cache, max_video_in_per_swap
            )
            
            # Se la cache è vuota o non ci sono candidati, passa alla prossima
            if not videos_in_cache or not candidates:
                continue

            for video_out in videos_in_cache:
                if swap_found_this_pass:  # Se abbiamo già trovato uno swap, esci
                    break
                
                for video_in in candidates:
                    # Verifica validità dello swap
                    if not is_swap_valid(video_out, video_in, cache_id, used_space, data, assignments):
                        continue

                    # Calcolo Delta + eventuale seconda best latency map
                    delta, second_best_map = calculate_swap_delta(video_out, video_in, cache_id, data, best_latency, assignments)

                    # Applicazione Swap
                    if delta > 0:
                        update_state_after_swap(video_out, video_in, cache_id, data, best_latency, assignments, second_best_map)
                        
                        total_saved_acc += delta
                        current_score = int((total_saved_acc / data.total_requests) * SCORE_MULTIPLIER)

                        swap_found_this_pass = True
                        total_swaps += 1
                        consecutive_successful_swaps += 1  # Incrementa contatore swap consecutivi

                        # Salvo info per stamparla a fine pass
                        last_swap_info = (video_out, video_in, cache_id, current_score)

                        break # Esce dal ciclo video_in

        # Se abbiamo fatto uno swap, chiudiamo l'iterator e ricominciamo dal while (nuovo pass)
        if swap_found_this_pass:
            iterator.close()
            passes_without_improvement = 0  # Reset contatore!
            if last_swap_info:
                video_out, video_in, cache_id, score = last_swap_info
                print(f"Pass {pass_count} completed. Swap [{video_out} out] -> [{video_in} in] | (Cache {cache_id}), Score: {score} | Consecutive swaps: {consecutive_successful_swaps}")
            continue  # Ricomincia il while → nuovo pass con cache_order randomizzato

        # Nessun swap trovato in questo pass
        passes_without_improvement += 1
        consecutive_successful_swaps = 0  # ✅ FIX: Reset contatore swap consecutivi dopo pass fallito
        iterator.close()
        print(f"Pass {pass_count} completed. No improvement. Consecutive passes without improvement: {passes_without_improvement}/{max_passes_no_improvement}. Score: {current_score}")
    
    # Determina motivo terminazione
    if consecutive_successful_swaps >= max_successful_swaps:
        print(f"\nLocal Search terminated: reached max consecutive successful swaps ({consecutive_successful_swaps}/{max_successful_swaps})")
    else:
        print(f"\nLocal Search terminated: no improvement for {max_passes_no_improvement} consecutive passes")
    print(f"Total: {pass_count} passes, {total_swaps} total swaps.")
    return assignments, current_score

