"""
Script per generare grafici per la presentazione dell'esame.
Analizza i risultati degli esperimenti e crea visualizzazioni comparative.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Imposta stile grafici
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11

# ============================================================================
# DATI SPERIMENTALI (estratti dai file spec)
# ============================================================================

data = {
    'me_at_the_zoo': {
        'K=1': {'greedy': 402352, 'final': 475634, 'time': 0.07, 'swaps': 32},
        'K=2': {'greedy': 447479, 'final': 495799, 'time': 0.07, 'swaps': 37},
        'K=3': {'greedy': 427839, 'final': 483732, 'time': 0.06, 'swaps': 29},
        'Marginal': {'greedy': 507906, 'final': 507906, 'time': 0.11, 'swaps': 0}
    },
    'trending_today': {
        'K=1': {'greedy': 499966, 'final': 499966, 'time': 22.4, 'swaps': 0},
        'K=2': {'greedy': 378343, 'final': 432597, 'time': 94.9, 'swaps': 2490},
        'K=3': {'greedy': 310654, 'final': 380119, 'time': 35.6, 'swaps': 2520},
        'Marginal': {'greedy': 499966, 'final': 499966, 'time': 22.6, 'swaps': 0}
    },
    'videos_worth_spreading': {
        'K=1': {'greedy': 176888, 'final': 475147, 'time': 5.4, 'swaps': 5000},
        'K=2': {'greedy': 224801, 'final': 425071, 'time': 5.0, 'swaps': 5000},
        'K=3': {'greedy': 266700, 'final': 430858, 'time': 5.1, 'swaps': 5000},
        'Marginal': {'greedy': 608277, 'final': 608946, 'time': 20.8, 'swaps': 342}
    },
    'kittens': {
        'K=1': {'greedy': 1008497, 'final': 1012920, 'time': 13500, 'swaps': 628},
        'K=2': {'greedy': 777220, 'final': 866121, 'time': 1347, 'swaps': 5000},
        'K=3': {'greedy': 657000, 'final': 744968, 'time': 611, 'swaps': 5000},
        'Marginal': {'greedy': 1021680, 'final': 1021709, 'time': 10569, 'swaps': 323}
    }
}

# Dati dettagliati aggiuntivi per analisi approfondita
detailed_stats = {
    'me_at_the_zoo': {
        'total_videos': 100,
        'total_caches': 10,
        'cache_capacity': 100,
        'K=1': {
            'heap_pops': 183,
            'heap_reinsertions': 0,
            'videos_cached': 32,
            'placements': 32,
            'space_util': 76.2,
            'termination': '5 passes no improvement'
        },
        'K=2': {
            'heap_pops': 183,
            'heap_reinsertions': 0,
            'videos_cached': 27,
            'placements': 47,
            'space_util': 93.3,
            'termination': '5 passes no improvement'
        },
        'K=3': {
            'heap_pops': 183,
            'heap_reinsertions': 0,
            'videos_cached': 25,
            'placements': 56,
            'space_util': 94.1,
            'termination': '5 passes no improvement'
        },
        'Marginal': {
            'heap_pops': 250,
            'heap_reinsertions': 67,
            'videos_cached': 29,
            'placements': 51,
            'space_util': 93.6,
            'termination': '5 passes no improvement'
        }
    },
    'trending_today': {
        'total_videos': 10000,
        'total_caches': 100,
        'cache_capacity': 50000,
        'K=1': {
            'heap_pops': 1000000,
            'heap_reinsertions': 0,
            'videos_cached': 9993,
            'placements': 9993,
            'space_util': 99.9,
            'termination': '5 passes no improvement'
        },
        'K=2': {
            'heap_pops': 1000000,
            'heap_reinsertions': 0,
            'videos_cached': 6893,
            'placements': 13786,
            'space_util': 99.9,
            'termination': '5 passes no improvement'
        },
        'K=3': {
            'heap_pops': 1000000,
            'heap_reinsertions': 0,
            'videos_cached': 5527,
            'placements': 16385,
            'space_util': 99.9,
            'termination': '5 passes no improvement'
        },
        'Marginal': {
            'heap_pops': 1000000,
            'heap_reinsertions': 0,
            'videos_cached': 9993,
            'placements': 9993,
            'space_util': 99.9,
            'termination': '5 passes no improvement'
        }
    },
    'videos_worth_spreading': {
        'total_videos': 10000,
        'total_caches': 100,
        'cache_capacity': 10000,
        'K=1': {
            'heap_pops': 135023,
            'heap_reinsertions': 0,
            'videos_cached': 4274,
            'placements': 4274,
            'space_util': 99.7,
            'termination': 'Max 5000 swaps'
        },
        'K=2': {
            'heap_pops': 135023,
            'heap_reinsertions': 0,
            'videos_cached': 3135,
            'placements': 5619,
            'space_util': 99.9,
            'termination': 'Max 5000 swaps'
        },
        'K=3': {
            'heap_pops': 135023,
            'heap_reinsertions': 0,
            'videos_cached': 2666,
            'placements': 6544,
            'space_util': 99.9,
            'termination': 'Max 5000 swaps'
        },
        'Marginal': {
            'heap_pops': 170501,
            'heap_reinsertions': 35478,
            'videos_cached': 2279,
            'placements': 6110,
            'space_util': 100.0,
            'termination': '5 passes no improvement'
        }
    },
    'kittens': {
        'total_videos': 10000,
        'total_caches': 500,
        'cache_capacity': 6000,
        'K=1': {
            'heap_pops': 4999996,
            'heap_reinsertions': 0,
            'videos_cached': 7475,
            'placements': 7475,
            'space_util': 97.6,
            'termination': '5 passes no improvement'
        },
        'K=2': {
            'heap_pops': 4999996,
            'heap_reinsertions': 0,
            'videos_cached': 5331,
            'placements': 10503,
            'space_util': 98.4,
            'termination': 'Max 5000 swaps'
        },
        'K=3': {
            'heap_pops': 4999996,
            'heap_reinsertions': 0,
            'videos_cached': 4395,
            'placements': 12887,
            'space_util': 98.7,
            'termination': 'Max 5000 swaps'
        },
        'Marginal': {
            'heap_pops': 9487229,
            'heap_reinsertions': 4487233,
            'videos_cached': 7407,
            'placements': 10815,
            'space_util': 100.0,
            'termination': '5 passes no improvement'
        }
    }
}


# Nomi dataset user-friendly
dataset_names = {
    'me_at_the_zoo': 'Me at the Zoo\n(Small)',
    'trending_today': 'Trending Today\n(Medium)',
    'videos_worth_spreading': 'Videos Worth\nSpreading\n(Complex)',
    'kittens': 'Kittens\n(Large)'
}

algorithms = ['K=1', 'K=2', 'K=3', 'Marginal']
colors = {
    'K=1': '#3498db',      # Blu
    'K=2': '#e74c3c',      # Rosso
    'K=3': '#f39c12',      # Arancione
    'Marginal': '#2ecc71'  # Verde
}

# ============================================================================
# GRAFICO 1: Score Finale per Algoritmo e Dataset
# ============================================================================

def plot_final_scores():
    """Bar chart raggruppato: score finale per algoritmo e dataset."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    datasets = list(data.keys())
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, algo in enumerate(algorithms):
        scores = [data[ds][algo]['final'] for ds in datasets]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, scores, width, label=algo, 
                      color=colors[algo], alpha=0.85, edgecolor='black', linewidth=0.7)
        
        # Aggiungi valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height/1000)}K',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score Finale (punti)', fontsize=13, fontweight='bold')
    ax.set_title('Confronto Score Finale per Algoritmo e Dataset', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_names[ds] for ds in datasets])
    ax.legend(title='Algoritmo', fontsize=11, title_fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grafico_1_score_finale.png', dpi=300, bbox_inches='tight')
    print("✓ Grafico 1 salvato: grafico_1_score_finale.png")
    plt.close()


# ============================================================================
# GRAFICO 2: Tempo di Esecuzione per Algoritmo e Dataset
# ============================================================================

def plot_execution_time():
    """Bar chart raggruppato: tempo di esecuzione (scala logaritmica)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    datasets = list(data.keys())
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, algo in enumerate(algorithms):
        times = [data[ds][algo]['time'] for ds in datasets]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, times, width, label=algo, 
                      color=colors[algo], alpha=0.85, edgecolor='black', linewidth=0.7)
        
        # Aggiungi valori sopra le barre
        for bar, time in zip(bars, times):
            height = bar.get_height()
            if time < 1:
                label = f'{time:.2f}s'
            elif time < 60:
                label = f'{time:.1f}s'
            elif time < 3600:
                label = f'{time/60:.1f}m'
            else:
                label = f'{time/3600:.1f}h'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Tempo di Esecuzione (secondi, scala log)', fontsize=13, fontweight='bold')
    ax.set_title('Confronto Tempo di Esecuzione per Algoritmo e Dataset', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_names[ds] for ds in datasets])
    ax.set_yscale('log')
    ax.legend(title='Algoritmo', fontsize=11, title_fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('grafico_2_tempo_esecuzione.png', dpi=300, bbox_inches='tight')
    print("✓ Grafico 2 salvato: grafico_2_tempo_esecuzione.png")
    plt.close()


# ============================================================================
# GRAFICO 3: Incremento Local Search (Stacked Bar)
# ============================================================================

def plot_ls_improvement():
    """Stacked bar chart: Greedy score + LS improvement."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    datasets = list(data.keys())
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, algo in enumerate(algorithms):
        greedy_scores = [data[ds][algo]['greedy'] for ds in datasets]
        ls_improvements = [data[ds][algo]['final'] - data[ds][algo]['greedy'] for ds in datasets]
        offset = (i - 1.5) * width
        
        # Barra greedy (base)
        bars1 = ax.bar(x + offset, greedy_scores, width, 
                       label=f'{algo} (Greedy)' if i == 0 else "",
                       color=colors[algo], alpha=0.6, edgecolor='black', linewidth=0.7)
        
        # Barra LS improvement (sopra)
        bars2 = ax.bar(x + offset, ls_improvements, width, bottom=greedy_scores,
                       label=f'{algo} (LS)' if i == 0 else "",
                       color=colors[algo], alpha=1.0, edgecolor='black', linewidth=0.7,
                       hatch='//')
        
        # Aggiungi etichette
        for j, (bar, greedy, improvement) in enumerate(zip(bars1, greedy_scores, ls_improvements)):
            if improvement > 0:
                # Percentuale miglioramento
                pct = (improvement / greedy) * 100
                ax.text(bar.get_x() + bar.get_width()/2., greedy + improvement,
                       f'+{pct:.1f}%', ha='center', va='bottom', 
                       fontsize=8, fontweight='bold', color=colors[algo])
    
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score (punti)', fontsize=13, fontweight='bold')
    ax.set_title('Incremento Local Search: Greedy vs Finale', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_names[ds] for ds in datasets])
    
    # Legenda personalizzata
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[algo], alpha=0.6, edgecolor='black', label=f'{algo} (Greedy)')
        for algo in algorithms
    ] + [
        Patch(facecolor='gray', alpha=1.0, hatch='//', edgecolor='black', label='LS Improvement')
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grafico_3_ls_improvement.png', dpi=300, bbox_inches='tight')
    print("✓ Grafico 3 salvato: grafico_3_ls_improvement.png")
    plt.close()


# ============================================================================
# GRAFICO 4: Score vs Tempo (Trade-off)
# ============================================================================

def plot_score_vs_time():
    """Scatter plot: score finale vs tempo (trade-off qualità/velocità)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for algo in algorithms:
        scores = []
        times = []
        dataset_labels = []
        
        for ds in data.keys():
            scores.append(data[ds][algo]['final'])
            times.append(data[ds][algo]['time'])
            dataset_labels.append(ds)
        
        ax.scatter(times, scores, label=algo, color=colors[algo], 
                  s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Aggiungi etichette dataset
        for i, ds in enumerate(dataset_labels):
            if ds != 'kittens':  # Per kittens le etichette si sovrappongono
                ax.annotate(ds.replace('_', ' ').title()[:10], 
                           (times[i], scores[i]),
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Tempo di Esecuzione (secondi, scala log)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score Finale (punti)', fontsize=13, fontweight='bold')
    ax.set_title('Trade-off Score vs Tempo per Algoritmo', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.legend(title='Algoritmo', fontsize=11, title_fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('grafico_4_score_vs_tempo.png', dpi=300, bbox_inches='tight')
    print("✓ Grafico 4 salvato: grafico_4_score_vs_tempo.png")
    plt.close()


# ============================================================================
# GRAFICO 5: Curva K vs Score (per dataset)
# ============================================================================

def plot_k_curve():
    """Line plot: K vs Score finale per ogni dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    k_values = [1, 2, 3]
    
    for idx, ds in enumerate(data.keys()):
        ax = axes[idx]
        
        # Estrai score per K=1,2,3
        scores_k = [data[ds][f'K={k}']['final'] for k in k_values]
        score_marginal = data[ds]['Marginal']['final']
        
        # Plot curva K
        ax.plot(k_values, scores_k, marker='o', linewidth=2.5, 
               markersize=10, color='#3498db', label='Static K')
        
        # Linea orizzontale marginal
        ax.axhline(y=score_marginal, color='#2ecc71', linestyle='--', 
                  linewidth=2, label='Marginal', alpha=0.8)
        
        # Evidenzia il massimo
        max_idx = np.argmax(scores_k)
        ax.scatter([k_values[max_idx]], [scores_k[max_idx]], 
                  color='red', s=200, zorder=5, marker='*', 
                  label=f'Best K={k_values[max_idx]}')
        
        ax.set_xlabel('K (copie max per video)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score Finale', fontsize=11, fontweight='bold')
        ax.set_title(dataset_names[ds], fontsize=12, fontweight='bold')
        ax.set_xticks(k_values)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Effetto del Parametro K sullo Score Finale', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('grafico_5_curva_k.png', dpi=300, bbox_inches='tight')
    print("✓ Grafico 5 salvato: grafico_5_curva_k.png")
    plt.close()


# ============================================================================
# GRAFICO 6: Totali Aggregati
# ============================================================================

def plot_total_scores():
    """Bar chart: score totale aggregato su tutti i dataset."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    totals = {}
    for algo in algorithms:
        totals[algo] = sum(data[ds][algo]['final'] for ds in data.keys())
    
    bars = ax.bar(totals.keys(), totals.values(), 
                  color=[colors[algo] for algo in algorithms],
                  alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Aggiungi valori e percentuali
    max_total = max(totals.values())
    for bar, algo in zip(bars, algorithms):
        height = bar.get_height()
        pct = (height / max_total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height/1000)}K\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Score Totale (4 dataset)', fontsize=13, fontweight='bold')
    ax.set_title('Confronto Score Totale Aggregato', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Aggiungi linea obiettivo Hash Code 2017
    ax.axhline(y=2651999, color='red', linestyle='--', linewidth=2, 
              label='Vincitore Hash Code 2017', alpha=0.7)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('grafico_6_totali.png', dpi=300, bbox_inches='tight')
    print("✓ Grafico 6 salvato: grafico_6_totali.png")
    plt.close()


# ============================================================================
# GRAFICI DETTAGLIATI PER DATASET (con statistiche heap, space util, LS)
# ============================================================================

def plot_detailed_dataset_analysis():
    """Crea grafici dettagliati per ogni dataset con tutte le statistiche."""
    
    for ds_key, ds_name in dataset_names.items():
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # --- Subplot 1: Heap Statistics (solo Marginal) ---
        ax1 = fig.add_subplot(gs[0, 0])
        marginal_stats = detailed_stats[ds_key]['Marginal']
        
        if marginal_stats['heap_reinsertions'] > 0:
            heap_data = [
                marginal_stats['heap_pops'] - marginal_stats['heap_reinsertions'],
                marginal_stats['heap_reinsertions']
            ]
            labels = ['Pops Iniziali', 'Re-insertions']
            colors_heap = ['#3498db', '#e74c3c']
            
            wedges, texts, autotexts = ax1.pie(heap_data, labels=labels, colors=colors_heap,
                                                autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            re_insert_rate = (marginal_stats['heap_reinsertions'] / marginal_stats['heap_pops']) * 100
            ax1.set_title(f'Heap Statistics - Marginal\nRe-insertion Rate: {re_insert_rate:.1f}%',
                         fontweight='bold', fontsize=11)
        else:
            ax1.text(0.5, 0.5, f"Heap Pops: {marginal_stats['heap_pops']:,}\nNo Re-insertions\n(Optimal immediato)",
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            ax1.set_title('Heap Statistics - Marginal', fontweight='bold', fontsize=11)
        
        # --- Subplot 2: Space Utilization ---
        ax2 = fig.add_subplot(gs[0, 1])
        space_utils = [detailed_stats[ds_key][algo]['space_util'] for algo in algorithms]
        bars = ax2.bar(algorithms, space_utils, color=[colors[algo] for algo in algorithms],
                      alpha=0.85, edgecolor='black', linewidth=1)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('% Capacità Usata', fontsize=11, fontweight='bold')
        ax2.set_title('Space Utilization', fontweight='bold', fontsize=11)
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3)
        
        # --- Subplot 3: Videos Cached vs Totali ---
        ax3 = fig.add_subplot(gs[0, 2])
        total_videos = detailed_stats[ds_key]['total_videos']
        videos_cached = [detailed_stats[ds_key][algo]['videos_cached'] for algo in algorithms]
        
        x_pos = np.arange(len(algorithms))
        bars1 = ax3.bar(x_pos, videos_cached, color=[colors[algo] for algo in algorithms],
                       alpha=0.85, edgecolor='black', linewidth=1)
        
        for bar, cached in zip(bars1, videos_cached):
            pct = (cached / total_videos) * 100
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{cached}\n({pct:.1f}%)', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        
        ax3.axhline(y=total_videos, color='red', linestyle='--', linewidth=1.5,
                   label=f'Totale: {total_videos}', alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(algorithms)
        ax3.set_ylabel('Video Cached', fontsize=11, fontweight='bold')
        ax3.set_title('Video Cached vs Totali', fontweight='bold', fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        
        # --- Subplot 4: Total Placements ---
        ax4 = fig.add_subplot(gs[1, 0])
        placements = [detailed_stats[ds_key][algo]['placements'] for algo in algorithms]
        
        bars2 = ax4.bar(algorithms, placements, color=[colors[algo] for algo in algorithms],
                       alpha=0.85, edgecolor='black', linewidth=1)
        
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        ax4.set_ylabel('Placements Totali', fontsize=11, fontweight='bold')
        ax4.set_title('Numero Placements (Video × Cache)', fontweight='bold', fontsize=11)
        ax4.grid(axis='y', alpha=0.3)
        
        # --- Subplot 5: Local Search Swaps ---
        ax5 = fig.add_subplot(gs[1, 1])
        swaps = [data[ds_key][algo]['swaps'] for algo in algorithms]
        
        bars3 = ax5.bar(algorithms, swaps, color=[colors[algo] for algo in algorithms],
                       alpha=0.85, edgecolor='black', linewidth=1)
        
        for bar, swap_count, algo in zip(bars3, swaps, algorithms):
            height = bar.get_height()
            termination = detailed_stats[ds_key][algo]['termination']
            
            if height > 0:
                label = f'{int(height)}\n{termination}'
            else:
                label = 'No LS\nneeded'
            
            ax5.text(bar.get_x() + bar.get_width()/2., height + max(swaps)*0.02,
                    label, ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
        
        ax5.set_ylabel('Total Swaps', fontsize=11, fontweight='bold')
        ax5.set_title('Local Search Dynamics', fontweight='bold', fontsize=11)
        ax5.grid(axis='y', alpha=0.3)
        
        # --- Subplot 6: Greedy vs Final Score ---
        ax6 = fig.add_subplot(gs[1, 2])
        x_pos = np.arange(len(algorithms))
        greedy_scores = [data[ds_key][algo]['greedy'] for algo in algorithms]
        improvements = [data[ds_key][algo]['final'] - data[ds_key][algo]['greedy'] for algo in algorithms]
        
        bars_g = ax6.bar(x_pos, greedy_scores, color=[colors[algo] for algo in algorithms],
                        alpha=0.6, edgecolor='black', linewidth=1, label='Greedy')
        bars_ls = ax6.bar(x_pos, improvements, bottom=greedy_scores,
                         color=[colors[algo] for algo in algorithms],
                         alpha=1.0, edgecolor='black', linewidth=1, hatch='//', label='LS Gain')
        
        for i, (greedy, improvement) in enumerate(zip(greedy_scores, improvements)):
            if improvement > 0:
                pct = (improvement / greedy) * 100
                ax6.text(i, greedy + improvement, f'+{pct:.1f}%',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(algorithms)
        ax6.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax6.set_title('Greedy vs Final Score', fontweight='bold', fontsize=11)
        ax6.legend(fontsize=9)
        ax6.grid(axis='y', alpha=0.3)
        
        # --- Subplot 7: Cache Occupancy (% caches utilizzate) ---
        ax7 = fig.add_subplot(gs[2, 0])
        total_caches = detailed_stats[ds_key]['total_caches']
        
        # Stima: ogni placement usa 1 cache, ma alcuni video sono in più cache
        # Approssimazione: caches_used ≈ placements / videos_cached (media placements per cache)
        cache_occupancy = []
        for algo in algorithms:
            placements_count = detailed_stats[ds_key][algo]['placements']
            # Stima semplificata: ogni cache ha circa placements/total_caches video
            avg_videos_per_cache = placements_count / total_caches
            # Se avg > 1, assumiamo che quasi tutte le cache sono usate
            if avg_videos_per_cache >= 1:
                occupancy = min(100, (placements_count / total_caches) * 10)  # Scaling
            else:
                occupancy = (placements_count / total_caches) * 100
            cache_occupancy.append(min(100, occupancy))
        
        bars4 = ax7.bar(algorithms, cache_occupancy, color=[colors[algo] for algo in algorithms],
                       alpha=0.85, edgecolor='black', linewidth=1)
        
        for bar in bars4:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        ax7.set_ylabel('% Cache Utilizzate (stima)', fontsize=11, fontweight='bold')
        ax7.set_title(f'Cache Occupancy ({total_caches} cache tot.)', fontweight='bold', fontsize=11)
        ax7.set_ylim(0, 105)
        ax7.grid(axis='y', alpha=0.3)
        
        # --- Subplot 8: Efficienza (Score/Tempo) ---
        ax8 = fig.add_subplot(gs[2, 1])
        efficiency = [data[ds_key][algo]['final'] / data[ds_key][algo]['time'] for algo in algorithms]
        
        bars5 = ax8.bar(algorithms, efficiency, color=[colors[algo] for algo in algorithms],
                       alpha=0.85, edgecolor='black', linewidth=1)
        
        for bar in bars5:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', rotation=0)
        
        ax8.set_ylabel('Score / Secondo', fontsize=11, fontweight='bold')
        ax8.set_title('Efficienza (Qualità/Velocità)', fontweight='bold', fontsize=11)
        ax8.grid(axis='y', alpha=0.3)
        
        # --- Subplot 9: Riepilogo Statistiche ---
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        summary_text = f"""
RIEPILOGO DATASET
━━━━━━━━━━━━━━━━━━━
Dataset: {ds_name.replace(chr(10), ' ')}
Video totali: {total_videos:,}
Cache totali: {total_caches}
Capacity: {detailed_stats[ds_key]['cache_capacity']:,} MB

VINCITORE: {max(algorithms, key=lambda a: data[ds_key][a]['final'])}
Score max: {max(data[ds_key][algo]['final'] for algo in algorithms):,}

MARGINAL INSIGHTS:
• Heap Pops: {marginal_stats['heap_pops']:,}
• Re-insertions: {marginal_stats['heap_reinsertions']:,}
• Space Util: {marginal_stats['space_util']:.1f}%
• Placements: {marginal_stats['placements']:,}
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Titolo principale
        fig.suptitle(f'Analisi Dettagliata: {ds_name.replace(chr(10), " ")}',
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Salva grafico
        filename = f'immagini/grafico_dettagliato_{ds_key}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Grafico dettagliato salvato: {filename}")
        plt.close()


# ============================================================================
# MAIN: Genera tutti i grafici
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generazione Grafici per Presentazione Esame")
    print("=" * 60)
    
    # Grafici generali
    plot_final_scores()
    plot_execution_time()
    plot_ls_improvement()
    plot_score_vs_time()
    plot_k_curve()
    plot_total_scores()
    
    # Grafici dettagliati per dataset
    print("\n" + "-" * 60)
    print("Generazione grafici dettagliati per ogni dataset...")
    print("-" * 60)
    plot_detailed_dataset_analysis()
    
    print("\n" + "=" * 60)
    print("✅ Tutti i grafici sono stati generati con successo!")
    print("=" * 60)
    print("\nGrafici generali creati:")
    print("  1. grafico_1_score_finale.png")
    print("  2. grafico_2_tempo_esecuzione.png")
    print("  3. grafico_3_ls_improvement.png")
    print("  4. grafico_4_score_vs_tempo.png")
    print("  5. grafico_5_curva_k.png")
    print("  6. grafico_6_totali.png")
    print("\nGrafici dettagliati per dataset:")
    print("  7. immagini/grafico_dettagliato_me_at_the_zoo.png")
    print("  8. immagini/grafico_dettagliato_trending_today.png")
    print("  9. immagini/grafico_dettagliato_videos_worth_spreading.png")
    print("  10. immagini/grafico_dettagliato_kittens.png")
