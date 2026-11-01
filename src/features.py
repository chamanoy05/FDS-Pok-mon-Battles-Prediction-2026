import json
import numpy as np
import pandas as pd

def load_type_chart(filepath):
    """Gen 2 tur samaradorligi jadvalini lug'at ko'rinishida yuklaydi."""
    try:
        type_df = pd.read_csv(filepath, index_col=0)
        return type_df.to_dict(orient='index')
    except Exception:
        return {}

def extract_features(battle, type_chart={}):
    """Bitta jangning JSON obyektini xususiyatlar (features) vektoriga o'tkazadi."""
    features = {
        'battle_id': battle.get('battle_id')
    }
    
    if 'player_won' in battle:
        features['target'] = 1 if battle['player_won'] is True else 0

    p1_team = battle.get('p1_team_details', [])
    p2_lead = battle.get('p2_lead_details', {})
    
    for stat in ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']:
        vals = [p.get(stat, 0) for p in p1_team]
        features[f'p1_mean_{stat}'] = np.mean(vals) if vals else 0
        features[f'p1_max_{stat}'] = np.max(vals) if vals else 0

    if p1_team and p2_lead:
        features['lead_spe_diff'] = p1_team[0].get('base_spe', 0) - p2_lead.get('base_spe', 0)
        features['lead_hp_diff'] = p1_team[0].get('base_hp', 0) - p2_lead.get('base_hp', 0)

    timeline = battle.get('battle_timeline', [])
    features['total_turns_logged'] = len(timeline)
    
    if timeline:
        last_turn = timeline[-1]
        p1_state = last_turn.get('p1_pokemon_state', {})
        p2_state = last_turn.get('p2_pokemon_state', {})
        
        p1_hp = p1_state.get('hp_pct', 100) if p1_state else 0
        p2_hp = p2_state.get('hp_pct', 100) if p2_state else 0
        features['p1_final_hp'] = p1_hp
        features['p2_final_hp'] = p2_hp
        features['final_hp_diff'] = p1_hp - p2_hp
        
        p1_switches = sum(1 for turn in timeline if turn.get('p1_move_details') is None)
        p2_switches = sum(1 for turn in timeline if turn.get('p2_move_details') is None)
        features['p1_switch_count'] = p1_switches
        features['p2_switch_count'] = p2_switches
    else:
        features['final_hp_diff'] = 0

    return features

def load_dataset(filepath, type_chart):
    """JSONL faylini o'qib, uni pandas DataFrame formatiga o'tkazadi."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(extract_features(json.loads(line), type_chart))
    return pd.DataFrame(records)
