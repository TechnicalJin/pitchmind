"""
Build Master Players Table from Squad Data
===========================================
Parses the IPL 2026 squad markdown file and creates master_players.csv
"""

import os
import re
import pandas as pd

def parse_squad_file(filepath: str) -> pd.DataFrame:
    """
    Parse squad markdown file and extract player information.

    Returns DataFrame with columns:
    - player_name (original name from PDF)
    - normalized_name (lowercase, no dots/hyphens)
    - team_2026
    - role
    - batting_style
    - bowling_style
    - player_type
    - is_overseas (inferred from name patterns)
    """

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Team patterns
    team_pattern = r'## \*\*\d+\. (.+?) \((.+?)\) — IPL 2026 Squad\*\*'

    # Split by team sections
    sections = re.split(r'## \*\*\d+\.', content)

    records = []
    current_team = None
    current_abbr = None

    for section in sections:
        # Extract team name
        team_match = re.search(r'(.+?) \((.+?)\) — IPL 2026 Squad\*\*', section)
        if team_match:
            current_team = team_match.group(1).strip()
            current_abbr = team_match.group(2).strip()

        if not current_team:
            continue

        # Parse table rows
        lines = section.split('\n')
        for line in lines:
            if not line.startswith('|'):
                continue
            if 'Player Name' in line or '---' in line:
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 5:
                continue

            # Extract fields: | Name | Role | Batting | Bowling | Type |
            name = parts[1].strip()
            role = parts[2].strip() if len(parts) > 2 else ""
            batting_style = parts[3].strip() if len(parts) > 3 else ""
            bowling_style = parts[4].strip() if len(parts) > 4 else ""
            player_type = parts[5].strip() if len(parts) > 5 else ""

            # Skip empty or header rows
            if not name or name.startswith('**') or name == 'Player Name':
                continue

            # Normalize name
            normalized = name.lower().strip().replace('.', '').replace('-', ' ')
            normalized = ' '.join(normalized.split())

            # Infer overseas status (basic heuristic)
            # Common overseas name patterns
            overseas_indicators = [
                'de kock', 'de villiers', 'du plessis', 'van der dussen',
                'bosch', 'brevis', 'miller', 'markram', 'rabada', 'ngidi',
                'buttler', 'stokes', 'archer', 'curran', 'livingstone',
                'head', 'cummins', 'starc', 'hazlewood', 'maxwell', 'warner',
                'williamson', 'boult', 'santner', 'phillips', 'neesham',
                'holder', 'russell', 'narine', 'bravo', 'hetmyer', 'pooran',
                'sammy', 'gayle', 'pollard', 'rashid khan', 'noor ahmad',
                'ghazanfar', 'omarzai', 'hasaranga', 'theekshana', 'mendis',
                'pathirana', 'thushara', 'ferguson', 'munro', 'guptill',
                'klaasen', 'nortje', 'jansen', 'stubbs', 'pretorius',
                'salt', 'david', 'jacks', 'bethell', 'duckett', 'topley',
                'overton', 'henry', 'ellis', 'short', 'green', 'inglis',
                'marsh', 'stoinis', 'jamieson', 'milne', 'burger', 'maphaka',
                'muzarabani', 'chameera', 'nissanka', 'carse', 'edwards',
                'banton', 'owen', 'bartlett', 'dwarshuis', 'connolly',
                'rickelton', 'rutherford'
            ]

            is_overseas = any(ind in normalized for ind in overseas_indicators)

            records.append({
                'player_name': name,
                'normalized_name': normalized,
                'team_2026': current_team,
                'team_abbr': current_abbr,
                'role': role,
                'batting_style': batting_style,
                'bowling_style': bowling_style,
                'player_type': player_type,
                'is_overseas': is_overseas
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    squad_file = "Squad_Data/IPL 2026 Squads — Quick Navigation 3263781c88e08017814dd5af05018b2e.md"

    if not os.path.exists(squad_file):
        print(f"Error: {squad_file} not found")
        exit(1)

    print("=" * 60)
    print("  Building Master Players Table")
    print("=" * 60)

    df = parse_squad_file(squad_file)

    print(f"\nParsed {len(df)} players from 10 teams\n")
    print("Team distribution:")
    print(df['team_abbr'].value_counts().to_string())

    print(f"\nOverseas players: {df['is_overseas'].sum()}")
    print(f"Indian players: {len(df) - df['is_overseas'].sum()}")

    # Save
    output_path = "data/master_players.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved to {output_path}")

    # Preview
    print("\nSample rows:")
    print(df[['player_name', 'team_abbr', 'role', 'is_overseas']].head(10).to_string(index=False))
