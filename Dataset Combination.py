#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
from difflib import get_close_matches
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

fbref_files = sorted(glob.glob("D:/#CSProject/Data/Fbref/*.csv"), reverse=True)  # Folder Location Containing Fbref files 
fbref_df = pd.concat([pd.read_csv(file, encoding='utf-8', encoding_errors='replace') for file in fbref_files], ignore_index=True)

football_data_files = sorted(glob.glob("D:/#CSProject/Data/FD/*.csv"), reverse=True)  # Folder Location Containing Football Data files 
football_data_df = pd.concat([pd.read_csv(file, encoding='utf-8', encoding_errors='replace') for file in football_data_files], ignore_index=True)


# In[2]:


fbref_df['Date'] = pd.to_datetime(fbref_df['Date'], errors='coerce')  
football_data_df['Date'] = pd.to_datetime(football_data_df['Date'], format='%d/%m/%Y', errors='coerce')  

fbref_df['Date'] = fbref_df['Date'].dt.strftime('%Y-%m-%d')
football_data_df['Date'] = football_data_df['Date'].dt.strftime('%Y-%m-%d')

football_data_df = football_data_df.iloc[:, :23]

fbref_df = fbref_df.rename(columns={
    'Home xG': 'Home_xG',
    'Away xG': 'Away_xG',
    'Home Team': 'Home_Team',
    'Away Team': 'Away_Team'
})

football_data_df = football_data_df.rename(columns={
    'HomeTeam': 'Home_Team',
    'AwayTeam': 'Away_Team',
    'Div': 'League'
})


# In[3]:


name_mapping_fbref = {
    'Spal': 'SPAL',
    'Arminia': 'Arminia Bielefeld',
    'Leeds United': 'Leeds',
    'Köln' : 'FC Koln',
    'Vitória' : 'Vitoria Guimaraes',
    'Paços' : 'Pacos Ferreira',
    'VitÃ³ria SetÃºbal' : 'Vitoria Setubal',
    'B-SAD' : 'Belenenses'
}

# Replace old names with new names in the 'Team' column
fbref_df['Home_Team'] = fbref_df['Home_Team'].replace(name_mapping_fbref)
fbref_df['Away_Team'] = fbref_df['Away_Team'].replace(name_mapping_fbref)

name_mapping_fd = {
    'Spal': 'SPAL',
    'Bielefeld': 'Arminia Bielefeld',
    'Man United': 'Manchester Utd',
    'Man City': 'Manchester City',
    'Guimaraes' : 'Vitoria Guimaraes',
    'Ath Bilbao' : 'Athletic Club',
    'Setubal' : 'Vitoria Setubal',
    'Sp Lisbon' : 'Sporting CP',
    'St. Gilloise' : 'Union SG'
}

football_data_df['Home_Team'] = football_data_df['Home_Team'].replace(name_mapping_fd)
football_data_df['Away_Team'] = football_data_df['Away_Team'].replace(name_mapping_fd)


# In[4]:


teams_fbref = set(fbref_df['Home_Team'].unique())
teams_football_data = set(football_data_df['Home_Team'].unique())

mapping = {}
for team in teams_fbref:
    closest_match = get_close_matches(team, teams_football_data, n=1)
    if closest_match:
        mapping[team] = closest_match[0]

# Apply the mapping to df1
fbref_df['Home_Team'] = fbref_df['Home_Team'].replace(mapping)
fbref_df['Away_Team'] = fbref_df['Away_Team'].replace(mapping)
football_data_df['Home_Team'] = football_data_df['Home_Team'].replace(mapping)
football_data_df['Away_Team'] = football_data_df['Away_Team'].replace(mapping)


# In[5]:


teams_fbref_conflict = set(fbref_df['Home_Team'].unique())
teams_football_data_conflict = set(football_data_df['Home_Team'].unique())

# Identify team names in df1 that are not in df2
conflicts_fbref = teams_fbref_conflict - teams_football_data_conflict

# Identify team names in df2 that are not in df1
conflicts_football_data = teams_football_data_conflict - teams_fbref_conflict

# Display conflicting names
print("Team names in fbref not in football data:", conflicts_fbref)
print("Team names in football data not in fbref:", conflicts_football_data)


# In[6]:


label_encoder = LabelEncoder()
football_data_df['FTR'] = label_encoder.fit_transform(football_data_df['FTR'])


# In[7]:


# Calculate the percentage of non-empty values in each column for fbref_df
fbref_non_empty = (fbref_df.select_dtypes(include=["number"]).notnull().sum() / len(fbref_df)) * 100
print("Percentage of non-empty values in fbref_df:")
print(fbref_non_empty)

# Calculate the percentage of non-empty values in each column for football_data_df
football_data_non_empty = (football_data_df.select_dtypes(include=["number"]).notnull().sum() / len(football_data_df)) * 100
print("\nPercentage of non-empty values in football_data_df:")
print(football_data_non_empty)


# In[8]:


football_data_df.select_dtypes(include=["number"]).corr()["FTR"].sort_values(ascending=False)


# In[9]:


football_data_df_copy = football_data_df.dropna()
# Select only numeric columns (excluding FTR for features)
X = football_data_df_copy.select_dtypes(include=["number"]).drop(columns=["FTR"])
y = football_data_df_copy["FTR"]
# Compute mutual information scores
mi_scores = mutual_info_classif(X, y)
# Convert to Pandas Series and sort
mi_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print(mi_results)


# In[10]:


football_data_df = football_data_df[['Date', 'League', 'Home_Team', 'Away_Team', 
                                     "FTR", "FTHG", "FTAG", "HS", "AS", "HST", "AST"]]
# Merge on Date, HomeTeam, and AwayTeam
combined_df = pd.merge(
    football_data_df,
    fbref_df[['Date', 'Home_Team', 'Away_Team', 'Home_xG', 'Away_xG']],  # Only select relevant columns
    on=['Date', 'Home_Team', 'Away_Team'],
    how='left'
)
combined_df = combined_df.dropna()


# In[11]:


def compute_gd_elo(df, k=32, initial_rating=1000):
    """
    Compute Goal-Based Elo ratings for each team before each match,
    storing and restoring Elo when a team returns to a league.
    Implements the method by Hvattum & Arntzen (2010).
    """
    teams = {}
    team_leagues = {}
    league_history = {}
    home_elo_list, away_elo_list = [], []

    def goal_multiplier(goal_diff):
        """Adjust K based on goal difference."""
        return 1 + (abs(goal_diff) ** 0.8)  # Scaling function

    for idx, row in df.iterrows():
        home_team, away_team = row['Home_Team'], row['Away_Team']
        home_goals, away_goals = row['FTHG'], row['FTAG']
        goal_diff = home_goals - away_goals  # Goal difference
        home_league, away_league = row['League'], row['League']

        # Initialize Elo if new team
        if home_team not in teams:
            teams[home_team] = initial_rating
        if away_team not in teams:
            teams[away_team] = initial_rating

        home_elo, away_elo = teams[home_team], teams[away_team]

        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)

        # Expected scores
        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1.0 - expected_home

        # Actual outcome
        if goal_diff > 0:
            score_home, score_away = 1.0, 0.0
        elif goal_diff < 0:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5

        # K-factor adjustment
        adjusted_k = k * goal_multiplier(goal_diff)

        # Update Elo ratings
        teams[home_team] = home_elo + adjusted_k * (score_home - expected_home)
        teams[away_team] = away_elo + adjusted_k * (score_away - expected_away)

    df['Home_ELO'] = home_elo_list
    df['Away_ELO'] = away_elo_list
    return df


# In[12]:


def compute_streaks(df):
    # Initialize dictionary to store streaks for each team
    team_streaks = {}

    # Lists to store streaks for the dataframe
    home_win_streaks, away_win_streaks = [], []
    home_loss_streaks, away_loss_streaks = [], []
    home_unbeaten_streaks, away_unbeaten_streaks = [], []
    home_winless_streaks, away_winless_streaks = [], []

    # Loop through each match in the dataframe
    for idx, row in df.iterrows():
        home_team, away_team = row['Home_Team'], row['Away_Team']
        ftr = row['FTR']  # FTR = 2 (Home Win), 0 (Away Win), 1 (Draw)

        # Initialize streaks if the team is new
        if home_team not in team_streaks:
            team_streaks[home_team] = {
                'home_win': 0, 'away_win': 0, 'home_loss': 0, 'away_loss': 0,
                'home_unbeaten': 0, 'away_unbeaten': 0, 'home_winless': 0, 'away_winless': 0
            }
        if away_team not in team_streaks:
            team_streaks[away_team] = {
                'home_win': 0, 'away_win': 0, 'home_loss': 0, 'away_loss': 0,
                'home_unbeaten': 0, 'away_unbeaten': 0, 'home_winless': 0, 'away_winless': 0
            }

        # Store current streaks before update
        home_win_streaks.append(team_streaks[home_team]['home_win'])
        away_win_streaks.append(team_streaks[away_team]['away_win'])
        home_loss_streaks.append(team_streaks[home_team]['home_loss'])
        away_loss_streaks.append(team_streaks[away_team]['away_loss'])
        home_unbeaten_streaks.append(team_streaks[home_team]['home_unbeaten'])
        away_unbeaten_streaks.append(team_streaks[away_team]['away_unbeaten'])
        home_winless_streaks.append(team_streaks[home_team]['home_winless'])
        away_winless_streaks.append(team_streaks[away_team]['away_winless'])

        # Update streaks based on result
        if ftr == 2:  # Home Win
            team_streaks[home_team]['home_win'] += 1
            team_streaks[away_team]['away_win'] = 0  # Reset away win streak

            team_streaks[home_team]['home_loss'] = 0  # Reset home loss streak
            team_streaks[away_team]['away_loss'] += 1  # Increase away loss streak

            team_streaks[home_team]['home_unbeaten'] += 1  # Increase home unbeaten streak
            team_streaks[away_team]['away_unbeaten'] = 0  # Reset away unbeaten streak

            team_streaks[home_team]['home_winless'] = 0  # Reset home winless streak
            team_streaks[away_team]['away_winless'] += 1  # Increase away winless streak

        elif ftr == 0:  # Away Win
            team_streaks[away_team]['away_win'] += 1
            team_streaks[home_team]['home_win'] = 0  # Reset home win streak

            team_streaks[away_team]['away_loss'] = 0  # Reset away loss streak
            team_streaks[home_team]['home_loss'] += 1  # Increase home loss streak

            team_streaks[away_team]['away_unbeaten'] += 1  # Increase away unbeaten streak
            team_streaks[home_team]['home_unbeaten'] = 0  # Reset home unbeaten streak

            team_streaks[away_team]['away_winless'] = 0  # Reset away winless streak
            team_streaks[home_team]['home_winless'] += 1  # Increase home winless streak

        else:  # Draw
            team_streaks[home_team]['home_win'] = 0  # Reset home win streak
            team_streaks[away_team]['away_win'] = 0  # Reset away win streak

            team_streaks[home_team]['home_loss'] = 0  # Reset home loss streak
            team_streaks[away_team]['away_loss'] = 0  # Reset away loss streak

            # Increase unbeaten streaks for both teams (separately tracked)
            team_streaks[home_team]['home_unbeaten'] += 1
            team_streaks[away_team]['away_unbeaten'] += 1

            # Increase winless streaks for both teams
            team_streaks[home_team]['home_winless'] += 1
            team_streaks[away_team]['away_winless'] += 1

    # Add calculated streaks to the dataframe
    df['Home_Win_Streak'] = home_win_streaks
    df['Away_Win_Streak'] = away_win_streaks
    df['Home_Loss_Streak'] = home_loss_streaks
    df['Away_Loss_Streak'] = away_loss_streaks
    df['Home_Unbeaten_Streak'] = home_unbeaten_streaks
    df['Away_Unbeaten_Streak'] = away_unbeaten_streaks
    df['Home_Winless_Streak'] = home_winless_streaks
    df['Away_Winless_Streak'] = away_winless_streaks

    return df


# In[13]:


def add_recent_points(df, window_sizes=[5, 10, 20]):
    """
    Adds rolling points over the last 'n' games for home and away teams.

    Parameters:
    df (pd.DataFrame): The match dataset with teams and results.
    window_sizes (list): The number of past games to consider (default: [5, 10]).

    Returns:
    pd.DataFrame: Updated dataframe with rolling points for each team.
    """
    team_points = {}  # Dictionary to track team results over time

    # Initialize columns for each rolling window
    for window in window_sizes:
        df[f"Home_Points_Last_{window}"] = 0
        df[f"Away_Points_Last_{window}"] = 0

    # Iterate over matches
    for idx, row in df.iterrows():
        home_team, away_team = row["Home_Team"], row["Away_Team"]
        ftr = row["FTR"]  # 2 = Home Win, 1 = Draw, 0 = Away Win

        # Assign points based on result
        home_points, away_points = 0, 0
        if ftr == 2:  # Home win
            home_points, away_points = 3, 0
        elif ftr == 1:  # Draw
            home_points, away_points = 1, 1
        elif ftr == 0:  # Away win
            home_points, away_points = 0, 3

        # Initialize history for teams if not present
        if home_team not in team_points:
            team_points[home_team] = []
        if away_team not in team_points:
            team_points[away_team] = []

        # Store rolling points before updating
        for window in window_sizes:
            df.at[idx, f"Home_Points_Last_{window}"] = sum(team_points[home_team][-window:])
            df.at[idx, f"Away_Points_Last_{window}"] = sum(team_points[away_team][-window:])

        # Append the latest match result to the team's history
        team_points[home_team].append(home_points)
        team_points[away_team].append(away_points)

    return df


# In[14]:


def add_net_stats(df):
    stat_pairs = {
        "FTG": ("FTHG", "FTAG"),
        "xG": ("Home_xG", "Away_xG"),
        "S": ("HS", "AS"),
        "ST": ("HST", "AST"),
    }

    # Loop through stat pairs and calculate net stats
    for stat_name, (home_stat, away_stat) in stat_pairs.items():
        df[f"Home_Net_{stat_name}"] = df[home_stat] - df[away_stat]
        df[f"Away_Net_{stat_name}"] = df[away_stat] - df[home_stat]

    return df


# In[15]:


def add_conceded_stats(df):
    df["FTHG_Conceded"] = df["FTAG"]
    df["FTAG_Conceded"] = df["FTHG"]

    df["HST_Conceded"] = df["AST"]
    df["AST_Conceded"] = df["HST"]
    
    df["HS_Conceded"] = df["AS"]
    df["AS_Conceded"] = df["HS"]
    
    df["Home_xG_Conceded"] = df["Away_xG"]
    df["Away_xG_Conceded"] = df["Home_xG"]
    
    return df


# In[16]:


combined_df = compute_gd_elo(combined_df)
combined_df = compute_streaks(combined_df)
combined_df = add_recent_points(combined_df)


# In[17]:


combined_df = add_net_stats(combined_df)
combined_df = add_conceded_stats(combined_df)


# In[18]:


correlations = combined_df.select_dtypes(include=["number"]).corr()["FTR"].sort_values(ascending=False)


# In[19]:


# Filter correlations that contain either "Streak" or "ELO"
engineered_stats_correlations = correlations[correlations.index.str.contains("Streak|Points")]

# Print the filtered correlations
print(engineered_stats_correlations)


# In[20]:


# Filter correlations that contain either "Streak" or "ELO"
engineered_stats_correlations2 = correlations[correlations.index.str.contains("Net|ELO|Conceded")]

# Print the filtered correlations
print(engineered_stats_correlations2)


# In[21]:


# Select only numeric columns (excluding FTR for features)
X = combined_df.select_dtypes(include=["number"]).drop(columns=["FTR"])
y = combined_df["FTR"]
# Compute mutual information scores
mi_scores = mutual_info_classif(X, y)
# Convert to Pandas Series and sort
mi_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)


# In[22]:


# Select only metrics that contain "Streak" or "ELO"
engineered_stats_mi = mi_results[mi_results.index.str.contains("Streak|Points")]

# Print the filtered metrics
print(engineered_stats_mi)


# In[23]:


# Select only metrics that contain "Streak" or "ELO"
engineered_stats_mi2 = mi_results[mi_results.index.str.contains("Net|ELO|Conceded")]

# Print the filtered metrics
print(engineered_stats_mi2)


# In[24]:


combined_df.to_csv("D:/#CSProject/Data/combined_datasetxg.csv", index=False)

