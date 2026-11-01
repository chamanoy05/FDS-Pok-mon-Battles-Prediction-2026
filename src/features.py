import json
import numpy as np
import pandas as pd

def load_type_chart(filepath):
    """
    Load the Gen 2 type effectiveness chart from a CSV file.
    Returns a nested dictionary mapping attacker type -> defender type -> multiplier.
    """
    try:
        type_df = pd.read_csv(filepath, index_col=0)
        return type_df.to_dict(orient='index')
    except Exception as e:
        # If type chart file is missing, print warning and return empty dict
        print(f"Warning: Could not load type chart from {filepath}: {e}")
        return {}

def calculate_type_advantage(atk_types, def_types, type_chart):
    """
    Calculate the overall type effectiveness multiplier of attacking types against defending types.
    """
    if not type_chart or not atk_types or not def_types:
        return 1.0
    
    total_multiplier = 1.0
    for a_type in atk_types:
        if a_type.lower() == 'notype':
            continue
        for d_type in def_types:
            if d_type.lower() == 'notype':
                continue
            # Get multiplier from type chart (defaults to neutral 1.0 if not found)
            mult = type_chart.get(a_type.lower(), {}).get(d_type.lower(), 1.0)
            total_multiplier *= mult
            
    return total_multiplier

def extract_features(battle, type_chart={}):
    """
    Extract high-predictive engineered features from a single battle record.
    Parses initial team stats, lead matchups, and 30-turn battle timeline dynamics.
    """
    features = {}
    
    # -------------------------------------------------------------------------
    # 1. CORE METADATA & TARGET
    # -------------------------------------------------------------------------
    features['battle_id'] = battle.get('battle_id')
    
    # Target variable: 1 if Player 1 won, 0 if lost (only in train.jsonl)
    if 'player_won' in battle:
        features['target'] = 1 if battle['player_won'] is True else 0

    # -------------------------------------------------------------------------
    # 2. INITIAL MATCHUP & LEAD ADVANTAGE (Static Features)
    # -------------------------------------------------------------------------
    p1_team = battle.get('p1_team_details', [])
    p2_lead = battle.get('p2_lead_details', {})
    
    # Aggregate base stats for Player 1's full team (6 Pokemon)
    for stat in ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']:
        stat_values = [p.get(stat, 0) for p in p1_team]
        features[f'p1_mean_{stat}'] = np.mean(stat_values) if stat_values else 0
        features[f'p1_max_{stat}'] = np.max(stat_values) if stat_values else 0
        features[f'p1_total_{stat}'] = np.sum(stat_values) if stat_values else 0

    # Lead vs Lead stat and type matchup differentials (Player 1 Lead is element 0)
    if p1_team and p2_lead:
        p1_lead = p1_team[0]
        
        # Speed differential is crucial in Pokemon to determine who moves first
        features['lead_speed_diff'] = p1_lead.get('base_spe', 0) - p2_lead.get('base_spe', 0)
        features['lead_hp_diff'] = p1_lead.get('base_hp', 0) - p2_lead.get('base_hp', 0)
        
        # Type effectiveness multipliers for the opening matchup
        p1_lead_types = p1_lead.get('types', [])
        p2_lead_types = p2_lead.get('types', [])
        
        features['p1_lead_type_adv'] = calculate_type_advantage(p1_lead_types, p2_lead_types, type_chart)
        features['p2_lead_type_adv'] = calculate_type_advantage(p2_lead_types, p1_lead_types, type_chart)
        features['lead_type_advantage_diff'] = features['p1_lead_type_adv'] - features['p2_lead_type_adv']

    # -------------------------------------------------------------------------
    # 3. BATTLE TIMELINE DYNAMICS (First 30 Turns)
    # -------------------------------------------------------------------------
    timeline = battle.get('battle_timeline', [])
    features['total_turns_logged'] = len(timeline)
    
    if timeline:
        last_turn = timeline[-1]
        p1_last_state = last_turn.get('p1_pokemon_state', {})
        p2_last_state = last_turn.get('p2_pokemon_state', {})
        
        # Final recorded HP percentages at turn 30 (or end of timeline)
        p1_final_hp = p1_last_state.get('hp_pct', 100) if p1_last_state else 0
        p2_final_hp = p2_last_state.get('hp_pct', 100) if p2_last_state else 0
        
        features['p1_final_hp'] = p1_final_hp
        features['p2_final_hp'] = p2_final_hp
        features['final_hp_diff'] = p1_final_hp - p2_final_hp
        
        # Status condition indicators at end of timeline (e.g., paralyze, sleep, freeze)
        features['p1_has_status'] = 1 if p1_last_state and p1_last_state.get('status') else 0
        features['p2_has_status'] = 1 if p2_last_state and p2_last_state.get('status') else 0
        
        # Track cumulative move power used and number of switches made across turns
        p1_total_power = 0
        p2_total_power = 0
        p1_switches = 0
        p2_switches = 0
        
        for turn in timeline:
            p1_move = turn.get('p1_move_details')
            p2_move = turn.get('p2_move_details')
            
            # If move_details is null, the player performed a Pokemon switch that turn
            if p1_move:
                p1_total_power += (p1_move.get('base_power', 0) or 0)
            else:
                p1_switches += 1
                
            if p2_move:
                p2_total_power += (p2_move.get('base_power', 0) or 0)
            else:
                p2_switches += 1

        features['p1_total_move_power'] = p1_total_power
        features['p2_total_move_power'] = p2_total_power
        features['move_power_diff'] = p1_total_power - p2_total_power
        features['p1_switch_count'] = p1_switches
        features['p2_switch_count'] = p2_switches
        features['switch_count_diff'] = p1_switches - p2_switches

    else:
        # Fallback values if timeline is empty
        features['p1_final_hp'] = 100
        features['p2_final_hp'] = 100
        features['final_hp_diff'] = 0
        features['p1_has_status'] = 0
        features['p2_has_status'] = 0
        features['p1_total_move_power'] = 0
        features['p2_total_move_power'] = 0
        features['move_power_diff'] = 0
        features['p1_switch_count'] = 0
        features['p2_switch_count'] = 0
        features['switch_count_diff'] = 0

    return features

def load_dataset(filepath, type_chart):
    """
    Parse a .jsonl dataset file line by line and convert it into a Pandas DataFrame.
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                # Extract features for each battle object
                battle_data = json.loads(line)
                records.append(extract_features(battle_data, type_chart))
                
    return pd.DataFrame(records)
