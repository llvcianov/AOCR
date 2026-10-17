import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Iterator
from tqdm import tqdm


FLOAT_TOLERANCE = 1e-6          # Tolleranza per confronto float nel calcolo dello score 
SCORE_MULTIPLIER = 1000          # Moltiplicatore per calcolo score
TQDM_UPDATE_INTERVAL = 1000     # Intervallo aggiornamento barre progresso

@dataclass
class ProblemData:
    # Dati di input
    V: int  # Numero di video
    E: int  # Numero di endpoint
    R: int  # Numero di richieste
    C: int  # Numero di cache
    X: int  # Capacità di ogni cache
    video_sizes: np.array # array contenente dimensione del video i-esimo

    # Lista di dizionari con informazioni sugli endpoint, strutturata così:
    # endpoint_id -> { 'datacenter_latency': int,
    #                  'cache_connections': [(cache_id, latency), ...]}
    endpoints: List[dict] #dict = dizionario generico in python. (Dict è un tipo specifico di dato, type hinting)
    requests_map: Dict[Tuple[int, int], int]            # (video, endpoint) -> n_requests
    cache_to_endpoints: Dict[int, List[int]]            # cache -> [endpoint_id1, ...]
    cache_endpoint_latency: Dict[Tuple[int, int], int]  # (cache, endpoint) -> latency
    cache_to_requested_videos: Dict[int, Set[int]]      # cache -> set di video id richiesti
    
    total_requests: int # Somma totale di tutte le richieste




def update_latencies_for_video_in_cache(video_id: int, cache_id: int, data, best_latency: Dict) -> float:
    """
    Aggiorna le best latencies per un video in una cache, ciclando su tutti gli endpoint connessi che richiedono quel video.
    
    INPUT:
        video_id: ID del video
        cache_id: ID della cache
        data: ProblemData
        best_latency: Dizionario da aggiornare 
    
    OUTPUT:
        Risparmio totale in millisecondi per questo video nella cache
    """
    total_savings = 0.0
    
    for endpoint_id in data.cache_to_endpoints[cache_id]:
        key = (video_id, endpoint_id)
        
        # Salta endpoint che non richiedono questo video.
        if key not in data.requests_map:
            continue
        
        cache_latency = data.cache_endpoint_latency[(cache_id, endpoint_id)]
        current_best = best_latency.get(key, data.endpoints[endpoint_id]['datacenter_latency'])
        
        # Se la cache è più veloce della best attuale
        if cache_latency < current_best:
            best_latency[key] = cache_latency
            # Calcola risparmio: (vecchia_best - nuova_best) * numero_richieste
            total_savings += (current_best - cache_latency) * data.requests_map[key]
    
    return total_savings


def is_swap_valid(video_out: int, video_in: int, cache_id: int, 
                  used_space: int, data, 
                  assignments: Dict[int, Set[int]]) -> bool:
    """
    Verifica se uno swap è valido secondo i vincoli del problema.
    
    INPUT:
        video_out: Video da rimuovere
        video_in: Video da aggiungere
        cache_id: Cache in cui fare lo swap
        used_space: Spazio attualmente usato nella cache
        data: Dati del problema
        assignments: Assegnamenti correnti
    
    OUTPUT:
        True se lo swap è valido, False altrimenti
    """
    # Video identico: non ha senso
    if video_in == video_out:
        return False
    
    # Video già presente nella cache
    if video_in in assignments[cache_id]:
        return False
    
    # Controllo capacità
    video_out_size = data.video_sizes[video_out]
    new_space = used_space + data.video_sizes[video_in] - video_out_size
    if new_space > data.X:
        return False
    
    return True


def get_stochastic_candidates(cache_id: int, assignments: Dict, data, 
                              max_out: int, max_in: int) -> Tuple[List[int], List[int], int]:
    """
    Seleziona casualmente un sottoinsieme di video presenti (OUT) e video richiesti (IN)
    per ridurre lo spazio di ricerca della Local Search.
    
    INPUT:
        cache_id: ID della cache da analizzare
        assignments: Dizionario cache_id -> set di video assegnati
        data: ProblemData con le strutture del problema
        max_out: Numero massimo di video OUT da selezionare
        max_in: Numero massimo di video IN da selezionare
    
    OUTPUT:
        Tuple contenente:
        - videos_out: Lista di video presenti nella cache (candidati alla rimozione)
        - videos_in: Lista di video richiesti ma non presenti (candidati all'inserimento)
        - used_space: Spazio attualmente occupato dalla cache
    """
    # Video Presenti (Candidati OUT)
    videos_in_cache_list = list(assignments[cache_id])
    if not videos_in_cache_list:
        return [], [], 0
        
    np.random.shuffle(videos_in_cache_list)
    videos_out = videos_in_cache_list[:max_out]

    # Spazio Usato 
    used_space = sum(data.video_sizes[v] for v in assignments[cache_id])

    # Video Richiesti (Candidati IN)
    candidates_set = data.cache_to_requested_videos[cache_id]
    if not candidates_set:
        return videos_out, [], used_space
        
    candidates_list = list(candidates_set)
    np.random.shuffle(candidates_list)
    videos_in = candidates_list[:max_in]

    return videos_out, videos_in, used_space


def find_second_best_latency(video_id: int, endpoint_id: int, 
                            exclude_cache: int, data, 
                            assignments: Dict[int, Set[int]]) -> int:
    """
    Trova la migliore latenza alternativa per un video da un endpoint,
    escludendo una specifica cache.
    
    INPUT:
        video_id: ID del video
        endpoint_id: ID dell'endpoint
        exclude_cache: Cache da escludere dalla ricerca
        data: Dati del problema
        assignments: Assegnamenti correnti
    
    OUTPUT:
        Latenza migliore alternativa (datacenter se nessuna cache ha il video)
    """
    # A. Partiamo dal caso peggiore: Data Center
    best_alternative = data.endpoints[endpoint_id]['datacenter_latency']
    
    # B. Controlliamo le altre cache connesse all'endpoint
    for cache_id, cache_latency in data.endpoints[endpoint_id]['cache_connections']:
        # Salta la cache esclusa
        if cache_id == exclude_cache:
            continue
        
        # Se la cache ha il video ed è più veloce dell'alternativa corrente
        if cache_id in assignments and video_id in assignments[cache_id]:
            if cache_latency < best_alternative:
                best_alternative = cache_latency
    
    return best_alternative


def _parse_endpoints(lines_iter: Iterator[str], E: int, pbar: tqdm) -> Tuple[List[dict], Dict, Dict]:
    """
    Legge la sezione degli Endpoint, costruendo le strutture dati necessarie. Aggiorna la barra di progresso.
    
    INPUT:
        lines_iter: Iteratore sulle righe del file
        E: Numero di endpoint da parsare
    
    OUTPUT:
        endpoints: Lista dizionari endpoint_id -> { 'datacenter_latency': int,
                                                    'cache_connections': [(cache_id, latency), ...]}

        cache_to_endpoints: dizionario cache -> [endpoint_id, ...]
        cache_endpoint_latency: dizionario (cache, endpoint) -> latency
    """
    endpoints = [] #lista vuota
    #defaultdict crea un dizionario che se non trova una chiave richiesta la inizializza con il valore di default fornito
    cache_to_endpoints = defaultdict(list) 
    cache_endpoint_latency = {} #dizionario vuoto

    #qui ciclo sulle righe di ogni endpoint (datacenter latency + numero connessioni cache)
    for endpoint_id in range(E):
        specs = next(lines_iter).split()
        dc_latency = int(specs[0])
        n_caches = int(specs[1])
        
        #lista connessioni cache per questo endpoint
        cache_connections = []
        #qui ciclo sui collegamenti cache di ogni endpoint (cache id + latency)
        for _ in range(n_caches):
            c_id, lat = map(int, next(lines_iter).split())
            cache_connections.append((c_id, lat))
            
            # Creo Strutture Dati
            cache_to_endpoints[c_id].append(endpoint_id)
            cache_endpoint_latency[(c_id, endpoint_id)] = lat
            
        endpoints.append({'datacenter_latency': dc_latency, 'cache_connections': cache_connections})
        pbar.update(1)
        
    return endpoints, cache_to_endpoints, cache_endpoint_latency


def _parse_requests(lines_iter: Iterator[str], R: int, endpoints: List[dict], pbar: tqdm) -> Tuple[Dict, Dict, int]:
    """
    Legge la sezione delle richieste e aggiorna la barra di progresso.
    
    INPUT:
        lines_iter: Iteratore sulle righe del file
        R: Numero di richieste da parsare
        endpoints: Lista dizionari endpoint_id -> { 'datacenter_latency': int,
                                                    'cache_connections': [(cache_id, latency), ...]}        
    
    OUTPUT:
        requests_map: Dizionario (video, endpoint) -> n_requests
        cache_to_requested_videos: Dizionario cache -> set di video id richiesti
        total_requests: Somma totale di tutte le richieste
    """
    requests_map = defaultdict(int)
    cache_to_requested_videos = defaultdict(set)
    total_requests = 0
    
    #ciclo su tutte le righe delle richieste
    for _ in range(R):
        v_id, e_id, num = map(int, next(lines_iter).split())
        
        requests_map[(v_id, e_id)] += num
        total_requests += num
        
        #accedo alle connessioni cache dell'endpoint associato alla richiesta
        #per ogni cache aggiorno il suo set di video richiesti
        for c_id, _ in endpoints[e_id]['cache_connections']:
            cache_to_requested_videos[c_id].add(v_id)
        
        pbar.update(1)
        
    return requests_map, cache_to_requested_videos, total_requests


def load_data(filename: str) -> ProblemData:
    """
    Legge file input e costruisce oggetto ProblemData
    
    INPUT:
        filename: Path del file di input
    
    OUTPUT:
        ProblemData: Oggetto contenente tutti i dati e strutture derivate
    """
    print(f"Reading {filename}...")

    #la funzione with si occupa di chiudere automaticamente il file dopo l'indentazione
    with open(filename, 'r') as f:
        #la funzione iter() crea un oggetto iteratore che restituisce una riga alla volta quando viene chiamato next()
        #la funzione strip() rimuove gli spazi bianchi iniziali e finali di ogni riga
        lines_iter = iter(line.strip() for line in f)
        
        "HEADER E VIDEO SIZES"
        # split() spezza la stringa "100 10 1000 5 500" in una lista ["100", "10", ...], 
        # map(int, ...) li converte tutti da testo a numero intero
        V, E, R, C, X = map(int, next(lines_iter).split())
        total_steps_bar = E + R

        with tqdm(total=total_steps_bar, desc="Parsing Data", unit="step") as pbar:

            #idem ma trasformo in array numpy per ottimizzare calcoli 
            video_sizes = np.array(list(map(int, next(lines_iter).split())))
            
            "ENDPOINTS"
            endpoints, c_to_e, c_e_lat = _parse_endpoints(lines_iter, E, pbar)
            
            "REQUESTS"
            req_map, c_req_vids, tot_req = _parse_requests(lines_iter, R, endpoints, pbar)
    
    return ProblemData(
        V, E, R, C, X, video_sizes, endpoints,
        req_map, c_to_e, c_e_lat, c_req_vids,
        tot_req
    )


def write_output(cache_assignments: Dict[int, Set[int]], filename: str):
    """
    Scrive il file di output nel formato richiesto 
    
    INPUT:
        cache_assignments: Dizionario cache_id -> set di video id assegnati
        filename: Path del file di output
    
    OUTPUT:
        None (scrive su file)
        
    FORMATO OUTPUT:
        Riga 1: N (numero cache utilizzate)
        Righe successive: <cache_id> <video_id_1> <video_id_2> ...
    """
    # Filtriamo le cache vuote e ordiniamo per ID
    valid_caches = []
    for cache_id, videos in cache_assignments.items():
        if videos:
            valid_caches.append((cache_id, sorted(list(videos))))
    
    #Ordiniamo le tuple in base all'ID delle cache
    # qui uso lambda function (lambda input: output)
    valid_caches.sort(key=lambda x: x[0])
    
    print(f"Writing output to {filename}...")

    with open(filename, 'w') as f:
        # Prima riga: Numero di cache usate
        f.write(f"{len(valid_caches)}\n")
        
        # Scriviamo le righe per ogni cache
        for cache_id, video_list in valid_caches:
            
            # Convertiamo la lista di numeri (video) in una lista di stringhe
            # Es: [3, 15] diventa ["3", "15"]
            video_strings = []
            for video in video_list:
                video_strings.append(str(video))
            
            # Uniamo le stringhe con uno spazio, Es: "3 15"
            line_content = " ".join(video_strings)
            f.write(f"{cache_id} {line_content}\n")

    print("Write complete.")


def print_stats(cache_assignments, cache_used_space, data):
    """
    Stampa le statistiche finali dell'esecuzione Greedy.
    """
    # Calcolo utilizzo spazio
    utilization = (cache_used_space.sum() / (data.C * data.X)) * 100
    
    # Calcolo video unici
    if cache_assignments:
        # è un set quindi non conta i duplicati
        all_unique = set()
        for videos in cache_assignments.values():
            #il metodo update aggiunge tutti gli elementi di 'videos' a 'all_unique'
            all_unique.update(videos)
        unique_count = len(all_unique)
    else:
        unique_count = 0
        
    # Calcolo totale placements
    total_placements = sum(len(v) for v in cache_assignments.values())
    print(
        f"Result: {unique_count}/{data.V} unique videos cached, "
        f"{total_placements} total placements, "
        f"{utilization:.1f}% space utilization"
    )




















