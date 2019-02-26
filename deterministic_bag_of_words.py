ACTION = 'action'
ADVENTURE = 'adventure'
ROLE_PLAYING = 'role playing'
SIMULATION = 'simulation'
STRATEGY = 'strategy'
SPORT = 'sport'
CASUAL = 'casual'
EDU = 'educational'
RELIGION = 'religion'
EXERECISE = 'exercise'
HORROR = 'horror'
OPEN_WORLD = 'open world'
MMO = 'mmo'

VIDEO = 'Video Games'
TABLE = 'Tabletop Games'
CARDS = "Playing Cards"


domain = [VIDEO, TABLE, CARDS]

categories = [ACTION, ADVENTURE, ROLE_PLAYING, SIMULATION, STRATEGY, SPORT, CASUAL, EDU, RELIGION, EXERECISE, HORROR, OPEN_WORLD, MMO]

subcategoreis = {ACTION: ['platform', 'shooter', 'fight', 'stealth', 'survival', 'rhythm', 'fps', 'tps'],
				 ADVENTURE: ['text adventure', 'graphic adventures', 'point and click', 'visual novel', 'interactive movie', 'Real-time 3D adventure'],
				 ROLE_PLAYING: ['RPG', 'MMORPG', 'ARPG', 'action RPG', 'Roguelikes', 'Tactical RPG', 'Sandbox RPG', 'Dungeon RPG', 'JRPG', 'Fantasy', 'Choices'],
				 SIMULATION: ['Construction', 'life', 'Vehicle', 'simulator'],
				 STRATEGY: ['rts', '4X', 'Artillery', 'Real-time strategy', 'Real-time tactics', 'rtt', 'moba', 'Multiplayer online battle arena', 'Tower defense', 'tbs', 'Turn based strategy', 'tbt', 'Turn-based tactics', 'Wargame'],
				 SPORT: ['racing']
				 }

DBoW = {y[i]: x for x, y in subcategoreis.items() for i in range(len(y))}
DBoW2 = {x:x for x in categories}
DBoW = {**DBoW2, **DBoW}

