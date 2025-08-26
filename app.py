import re
import random
from typing import Dict, List, Tuple
import spotipy

# ============ 改善 1: 更精確的文字分析 ============

def classify_mood_detailed(text: str) -> Dict:
    """
    更細緻的情境分析，返回多個維度的分數
    """
    text = text.lower()
    
    # 能量維度 (0-1)
    energy_keywords = {
        'high': ['派對', 'party', '嗨', '開趴', '運動', 'workout', '跳舞', 'dance', '興奮'],
        'low': ['放鬆', 'chill', '冷靜', '安靜', '睡覺', '休息', '疲累']
    }
    
    # 情緒維度 (0-1)
    mood_keywords = {
        'positive': ['開心', 'happy', '快樂', '爽', '慶祝', '興奮'],
        'negative': ['悲傷', 'sad', '難過', '失戀', '心碎', '憂鬱', '沮喪'],
        'neutral': ['專注', 'focus', '讀書', '工作', 'coding']
    }
    
    # 場景維度
    context_keywords = {
        'study': ['讀書', '學習', '專注', 'study', 'focus', 'coding', '工作'],
        'social': ['派對', 'party', '朋友', '聚會', '開趴'],
        'alone': ['獨處', '一個人', '深夜', '思考', '回憶'],
        'exercise': ['運動', 'workout', '跑步', '健身', '瑜伽']
    }
    
    # 音樂風格
    genre_hints = {
        'acoustic': ['吉他', '原聲', 'acoustic', '民謠', '清新'],
        'electronic': ['電音', 'edm', '電子', 'techno', 'house'],
        'jazz': ['爵士', 'jazz', '咖啡廳', '慵懶'],
        'classical': ['古典', '鋼琴', 'piano', '弦樂'],
        'lofi': ['lofi', 'lo-fi', '輕音樂']
    }
    
    def calculate_score(keywords_dict, default=0.5):
        scores = {}
        for category, keywords in keywords_dict.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[category] = min(1.0, score * 0.3 + default)
        return scores
    
    energy_scores = calculate_score(energy_keywords)
    mood_scores = calculate_score(mood_keywords)
    context_scores = calculate_score(context_keywords)
    genre_scores = calculate_score(genre_hints, default=0)
    
    # 計算最終參數
    energy = energy_scores.get('high', 0.5) - energy_scores.get('low', 0.5) + 0.5
    energy = max(0.1, min(0.9, energy))
    
    valence = mood_scores.get('positive', 0.5) - mood_scores.get('negative', 0.5) + 0.5
    valence = max(0.1, min(0.9, valence))
    
    # 根據場景調整其他參數
    danceability = 0.5
    acousticness = 0.3
    tempo = 110
    
    if context_scores.get('social', 0) > 0.5:
        danceability = 0.8
        tempo = 125
    elif context_scores.get('study', 0) > 0.5:
        energy = min(energy, 0.4)
        acousticness = 0.7
        tempo = 85
    elif context_scores.get('exercise', 0) > 0.5:
        energy = 0.9
        danceability = 0.9
        tempo = 140
    
    # 風格偏好
    preferred_genres = [genre for genre, score in genre_scores.items() if score > 0]
    
    return {
        'target_energy': energy,
        'target_valence': valence,
        'target_danceability': danceability,
        'target_acousticness': acousticness,
        'target_tempo': tempo,
        'preferred_genres': preferred_genres,
        'context': max(context_scores.keys(), key=lambda k: context_scores.get(k, 0))
    }

# ============ 改善 2: 智慧外部來源選擇 ============

def get_context_playlists(sp, context: str, preferred_genres: List[str]) -> List[str]:
    """
    根據情境和偏好風格選擇不同的 Spotify 歌單來源
    """
    playlist_sources = {
        'study': [
            'Deep Focus', 'Peaceful Piano', 'Lofi Hip Hop', 'Ambient Chill',
            'Brain Food', 'Instrumental Study', 'Coffee Table Jazz'
        ],
        'social': [
            'Party Hits', 'Dance Pop', 'Feel Good Pop', 'Party Anthems',
            'Dance Hits', 'Pop Rising', 'Mood Booster'
        ],
        'alone': [
            'Melancholy Indie', 'Sad Songs', 'Indie Folk', 'Contemplative',
            'Rainy Day', 'Solitude', 'Night Shift'
        ],
        'exercise': [
            'Workout', 'Power Workout', 'Cardio', 'Running',
            'Beast Mode', 'Workout Twerkout', 'Motivation Mix'
        ],
        'default': [
            'Today\'s Top Hits', 'Discover Weekly', 'Release Radar',
            'Fresh Finds', 'New Music Friday'
        ]
    }
    
    # 根據風格偏好添加特定歌單
    genre_playlists = {
        'jazz': ['Jazz Classics', 'Smooth Jazz', 'Jazz Vibes'],
        'acoustic': ['Acoustic Hits', 'Folk & Friends', 'Unplugged'],
        'electronic': ['Electronic Mix', 'Dance Hits', 'Electronic Rising'],
        'lofi': ['Lofi Hip Hop', 'Chill Lofi Study Beats', 'lofi hip hop radio']
    }
    
    target_names = playlist_sources.get(context, playlist_sources['default'])
    
    # 添加風格特定歌單
    for genre in preferred_genres:
        if genre in genre_playlists:
            target_names.extend(genre_playlists[genre])
    
    # 搜尋實際存在的歌單
    found_playlists = []
    for name in target_names:
        try:
            results = sp.search(q=name, type='playlist', limit=3)
            playlists = results.get('playlists', {}).get('items', [])
            for pl in playlists:
                if pl.get('id') and pl.get('tracks', {}).get('total', 0) > 20:
                    found_playlists.append(pl['id'])
                    if len(found_playlists) >= 8:  # 限制數量
                        return found_playlists
        except Exception as e:
            print(f"搜尋歌單 '{name}' 失敗: {e}")
            continue
    
    return found_playlists or ['37i9dQZF1DXcBWIGoYBM5M']  # fallback

# ============ 改善 3: 更好的相似度計算 ============

def calculate_track_similarity(track_features: Dict, target_params: Dict) -> float:
    """
    計算歌曲與目標參數的相似度，使用加權距離
    """
    if not track_features:
        return 0.0
    
    weights = {
        'energy': 0.25,
        'valence': 0.25,
        'danceability': 0.15,
        'acousticness': 0.15,
        'tempo': 0.20
    }
    
    similarity = 0.0
    total_weight = 0.0
    
    for feature, weight in weights.items():
        target_key = f'target_{feature}'
        if target_key in target_params and feature in track_features:
            target_val = target_params[target_key]
            track_val = track_features[feature]
            
            if feature == 'tempo':
                # tempo 需要正規化 (通常在 50-200 之間)
                diff = abs(track_val - target_val) / 150.0
            else:
                # 其他特徵都在 0-1 之間
                diff = abs(track_val - target_val)
            
            similarity += weight * (1.0 - diff)
            total_weight += weight
    
    return similarity / total_weight if total_weight > 0 else 0.0

# ============ 改善 4: 多樣性增強 ============

def ensure_diversity(tracks: List[Dict], max_per_artist: int = 2) -> List[Dict]:
    """
    確保歌單的多樣性：限制同一藝人的歌曲數量
    """
    artist_counts = {}
    diverse_tracks = []
    
    # 隨機打亂以增加變化
    shuffled_tracks = tracks.copy()
    random.shuffle(shuffled_tracks)
    
    for track in shuffled_tracks:
        artists = track.get('artists', [])
        if not artists:
            continue
            
        primary_artist = artists[0].get('name', 'Unknown')
        current_count = artist_counts.get(primary_artist, 0)
        
        if current_count < max_per_artist:
            diverse_tracks.append(track)
            artist_counts[primary_artist] = current_count + 1
            
            if len(diverse_tracks) >= 10:  # 限制總數
                break
    
    return diverse_tracks

# ============ 改善 5: 主要推薦函數 ============

def generate_smart_playlist(sp, user_input: str) -> Tuple[List[Dict], Dict]:
    """
    智慧生成歌單的主函數
    """
    # 1. 分析使用者輸入
    mood_analysis = classify_mood_detailed(user_input)
    
    # 2. 收集使用者音樂庫
    user_tracks = collect_user_tracks(sp, max_n=100)
    
    # 3. 根據情境收集外部音樂
    context = mood_analysis.get('context', 'default')
    preferred_genres = mood_analysis.get('preferred_genres', [])
    
    external_playlists = get_context_playlists(sp, context, preferred_genres)
    external_tracks = []
    
    for playlist_id in external_playlists:
        try:
            tracks = fetch_playlist_tracks(sp, playlist_id, max_n=50)
            external_tracks.extend(tracks)
            if len(external_tracks) >= 200:
                break
        except Exception as e:
            print(f"獲取歌單 {playlist_id} 失敗: {e}")
            continue
    
    # 4. 獲取音訊特徵
    all_track_ids = []
    for track in user_tracks + external_tracks:
        track_id = track.get('id')
        if track_id and len(track_id) == 22:
            all_track_ids.append(track_id)
    
    # 去重
    all_track_ids = list(set(all_track_ids))[:300]  # 限制數量避免 API 限制
    features_map = audio_features_map(sp, all_track_ids)
    
    # 5. 評分和排序
    def score_track(track):
        track_id = track.get('id')
        if not track_id or track_id not in features_map:
            return 0.0
        return calculate_track_similarity(features_map[track_id], mood_analysis)
    
    # 對使用者和外部音樂分別評分
    user_scored = [(score_track(t), t) for t in user_tracks]
    user_scored.sort(key=lambda x: x[0], reverse=True)
    
    external_scored = [(score_track(t), t) for t in external_tracks]
    external_scored.sort(key=lambda x: x[0], reverse=True)
    
    # 6. 選擇最終歌單 (3 familiar + 7 new)
    user_library_ids = {t.get('id') for t in user_tracks}
    
    # 選擇熟悉歌曲 (最多3首)
    familiar = []
    for score, track in user_scored[:10]:  # 從前10首中選
        if len(familiar) < 3 and track.get('id'):
            track['source'] = 'familiar'
            familiar.append(track)
    
    # 選擇新歌曲 (最多7首，排除使用者已有的)
    new_tracks = []
    for score, track in external_scored:
        if len(new_tracks) >= 7:
            break
        track_id = track.get('id')
        if track_id and track_id not in user_library_ids:
            track['source'] = 'new'
            new_tracks.append(track)
    
    # 7. 確保多樣性並混合
    final_playlist = familiar + new_tracks
    final_playlist = ensure_diversity(final_playlist, max_per_artist=2)
    
    # 8. 補足到10首 (如果不夠的話)
    if len(final_playlist) < 10:
        additional_externals = [t for s, t in external_scored[len(new_tracks):]]
        additional_diverse = ensure_diversity(additional_externals, max_per_artist=1)
        
        for track in additional_diverse:
            if len(final_playlist) >= 10:
                break
            if track.get('id') not in [t.get('id') for t in final_playlist]:
                track['source'] = 'additional'
                final_playlist.append(track)
    
    return final_playlist[:10], mood_analysis

# 在你的主要 recommend 函數中替換原本的邏輯：
def improved_recommend_logic(sp, text):
    """
    替換原本的推薦邏輯
    """
    try:
        playlist, analysis = generate_smart_playlist(sp, text)
        
        # 記錄分析結果 (用於除錯)
        print(f"分析結果: {analysis}")
        print(f"生成歌單: {len(playlist)} 首")
        familiar_count = len([t for t in playlist if t.get('source') == 'familiar'])
        new_count = len([t for t in playlist if t.get('source') == 'new'])
        print(f"熟悉: {familiar_count}, 新歌: {new_count}")
        
        return playlist
        
    except Exception as e:
        print(f"推薦邏輯錯誤: {e}")
        raise
