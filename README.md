# Maze Agent

[English](#english) · [Français](#français)

![The cat escaping level 3, its thoughts streamed live (sped up)](docs/demo.gif)

## English

An experimentation project with **LangGraph 1.2**. A cat controlled by an LLM has to escape six mazes of increasing difficulty. The game exists to exercise LangGraph's main features in one small, readable codebase: graph state and reducers, command-based routing, checkpointing, human-in-the-loop interrupts, long-term memory, streaming, runtime context, and node-level timeouts and retries.

### Stack

| Package | Version |
|---|---|
| Python | 3.12 |
| `langgraph` | 1.2.24 |
| `langgraph-checkpoint` | 4.3.1 |
| `langchain` / `langchain-core` | 1.5.2 / 1.8.4 |
| `langchain-openai` / `langchain-anthropic` | 1.8.5 / 1.7.12 |
| `fastapi` | 0.165.0 |
| `pydantic` | 2.14.2 |

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The frontend is plain HTML and JavaScript, with Mermaid loaded from a CDN.

### Getting started

```bash
cp .env.example .env              # set OPENAI_API_KEY, optionally ANTHROPIC_API_KEY
uv run uvicorn maze_agent.server:app
```

Open http://127.0.0.1:8000, choose a model and press **Play**.

| Command | Purpose |
|---|---|
| `uv run python -m maze_agent --level 5 --attempts 3` | Play a level in the terminal, retrying after failures |
| `uv run python -m maze_agent.eval --models <ids> --levels 1 2 3` | Run the evaluation harness |
| `uv run pytest` | Run the tests (no API key required) |

### The graph

```
START ──> think ──> act ──┬──> think        the cat keeps exploring
                          ├──> ask_human ──> think    the cat is stuck: wait for the player
                          ├──> reflect ──> END        the cat failed: write a lesson
                          └──> END                    the cat escaped
```

| Node | Kind | Role |
|---|---|---|
| `think` | async, LLM | Builds the cat's observation and asks the model for a structured `Decision` (a thought, then a direction or `wait`). |
| `act` | pure Python | Applies the move and the world rules, updates lives and status, then routes with `Command(goto=...)`. |
| `ask_human` | interrupt | Pauses the graph with `interrupt()` until the player sends a hint. |
| `reflect` | async, LLM | After a failure, writes one lesson into the long-term `Store`. |

#### What happens during a run

1. **Setup.** The level is generated from its seed, and the initial state (position, world, lives, visited cells) is created with a fresh `thread_id`.
2. **`think`.** The observation is built from the state (see *What the cat perceives*). Up to three lessons from previous failures on the same level and model are read from the `Store` and appended, along with any hint the player just gave. The model answers with a `Decision`; its token usage and cost are added to the state through reducers.
3. **`act`.** The move is applied: a wall or the locked door keeps the cat in place, the key is picked up, spikes and dogs cost a life. Dogs then take their turn. The status becomes `won`, `dead`, `out_of_steps` or stays `running`.
4. **Routing.** `act` returns a `Command` that sends the graph to:
   - `END` if the cat escaped;
   - `reflect` if it died or ran out of steps;
   - `ask_human` if it is stuck (see below) and a player is connected;
   - `think` otherwise.
5. **`ask_human`.** The cat is considered stuck when its mental map has no unexplored opening left, or when its last 8 positions cover 2 cells or fewer. A cooldown of 15 steps prevents repeated calls. The graph stops with `interrupt()`; the web server ends the stream and the player's answer resumes the same thread with `Command(resume=...)`. The hint is shown to the LLM on the next `think`.
6. **`reflect`.** The model receives a summary of the attempt (result, cells explored, key status, every injury with its step and position) and writes a lesson of at most 30 words, stored under the namespace `("lessons", "<level>:<model>")`.

### The game

#### Levels

| # | Name | Size | Spikes | Dogs | Door | Extra passages | Lives |
|---|---|---|---|---|---|---|---|
| 1 | First steps | 7×7 | | | | | 3 |
| 2 | Getting lost | 10×10 | | | | | 3 |
| 3 | Spikes | 12×12 | 8 | | | 3 | 3 |
| 4 | Dogs | 14×14 | 8 | 3 | | 8 | 3 |
| 5 | Locked door | 16×16 | 10 | 3 | ✓ | 10 | 3 |
| 6 | The final maze | 18×18 | 14 | 4 | ✓ | 14 | 5 |

Each level allows `2 × size²` steps.

#### Generation

- **Maze.** A seeded growing-tree algorithm carves a perfect maze, growing from the newest cell half of the time and from a random cell otherwise. This gives many junctions and short dead ends. A few extra walls are then opened to create loops, so dangers can be walked around.
- **Door and key.** The door sits three quarters of the way along the exit path. The key is placed before the door, on the cell farthest from both the start and the door. Loops never connect the two sides of the door, so it cannot be bypassed.
- **Spikes.** Placed off the exit path, as close to it as possible (side-corridor entrances first), never behind another spike and never on the way to the key.
- **Dogs.** Placed off the path, far enough from the start, in cells from which they can reach the exit path without crossing spikes.
- Every level can be solved without losing a life; the tests check this for all six.

#### Rules

- Each turn the cat moves one cell or waits.
- Spikes cost a life every time they are stepped on.
- Dogs move every other turn. They patrol forward and turn back only at dead ends; they never chase. They avoid spikes, the locked door and the start. A dog that bites goes back to its home cell.
- The door blocks everyone until the cat carries the key.

#### What the cat perceives

The observation sent to the LLM contains:

- its position, the exit, the step, its lives and whether it carries the key;
- what happened on the last turn (wall, locked door, spikes, bite, key);
- each open direction with the neighbouring cell, how often it was visited, and what is there (exit, dog, spikes, key, door), plus a warning when a dog stands right next to that cell;
- the dogs it can hear (within 3 cells) and whether each is coming closer or moving away;
- a **mental map** hint: the nearest unexplored opening reachable through visited cells, avoiding spikes and the locked door;
- the lessons from previous attempts and the player's hint, when there are any.

### LangGraph features used

| Feature | Where | Purpose |
|---|---|---|
| `StateGraph` with a Pydantic state | `agent.py` `State` | The state stays typed after a checkpoint reload. A plain dict would turn tuples into lists. |
| Reducers (`Annotated[..., add]`) | `State.tokens`, `cost`, `incidents` | Nodes return only their share; LangGraph sums tokens and cost and appends injuries. |
| Runtime context (`context_schema`, `Runtime[Context]`) | `Context` | Injects the maze, the LLM brain and run options. Tests inject a scripted brain. |
| Async nodes with `timeout` and `RetryPolicy` | `think`, `reflect` | A hanging LLM call is cancelled after 30 s and retried up to 3 times. |
| `Command(update=..., goto=...)` and `destinations` | `act` | One node updates the state and chooses the next node; `destinations` declares the possible targets for the diagram. |
| Checkpointer (`InMemorySaver`) | graph compile | Saves every step; required by interrupts. A `JsonPlusSerializer` allowlist declares the custom types it may restore. |
| `interrupt()` and `Command(resume=...)` | `ask_human`, `server.py` | Pauses a run and resumes the same thread with the player's answer. |
| Long-term memory (`InMemoryStore`, `runtime.store`) | `reflect`, `think` | Lessons persist across runs, one namespace per level and model. |
| Streaming (`astream`, `version="v2"`, `updates` + `messages`) | `runner.py` | `updates` drive the board; `messages` carries the LLM tokens of `think`, filtered by `langgraph_node`. |
| Graph introspection (`get_graph().draw_mermaid()`) | `/api/graph` | The compiled graph is rendered in the UI, and the running node is highlighted. |
| Run config (`tags`, `metadata`, `recursion_limit`) | `runner.py` | Labels each run with its level and model, for LangSmith when tracing is enabled. |

From LangChain, `init_chat_model` builds any provider's model from a single id, and `with_structured_output(method="json_schema")` constrains the answer to the `Decision` schema using the provider's native structured output.

### Web application

The FastAPI server streams runs to the browser with native Server-Sent Events (`fastapi.sse`).

| Endpoint | Description |
|---|---|
| `GET /api/levels` | Levels and available models |
| `GET /api/levels/{n}` | Maze and hazards of a level |
| `GET /api/levels/{n}/play?model=` | Starts a run (SSE) |
| `GET /api/runs/{thread_id}/resume?answer=` | Resumes a paused run with the player's hint (SSE) |
| `GET /api/lessons?level=&model=` / `DELETE` | Reads or clears the stored lessons |
| `GET /api/graph` | Mermaid source of the compiled graph |
| `GET /api/eval` | Latest evaluation report |

A run emits `start`, `token`, `think`, `act`, `interrupt`, `ask_human`, `reflect` and `done` events. Closing the page closes the stream, which cancels the graph and stops LLM calls.

The page shows the maze, the live graph, the cat's thoughts as they are generated (each one expandable to the exact prompt), the stored lessons, a help panel when the cat asks for a hint, and an **Evaluation** tab.

### Models

| Model | Input / output price per million tokens |
|---|---|
| `openai:gpt-5.4-mini` (default) | $0.75 / $4.50 |
| `openai:gpt-5.4-nano` | $0.20 / $1.25 |
| `anthropic:claude-haiku-5-5` | $0.10 / $0.50 |

A model can be selected only when its API key is set; otherwise it is greyed out in the picker. Token usage comes from each response, and the cost of a run is computed from this table.

### Evaluation

`maze_agent/eval.py` plays chains of attempts for every model and level. Each chain stops at its first win, and every attempt reads the lessons written by the failed ones. Each `(model, level)` pair reports:

- win rate on the first attempt, and win rate within the chain;
- average steps of the winning attempts, against the optimal path (fetching the key first when there is a door);
- average lives lost;
- average cost per attempt, and total cost per win.

The report is written to `evals/results.json` and displayed in the Evaluation tab. `--update` re-measures the requested models and levels and keeps the rest of the report.

### Tests

The tests use a scripted brain, so they run without an API key. They cover maze generation, level constraints, world rules, the mental map, the observation text, cost accounting, the lesson cycle through the `Store`, the interrupt and resume cycle (in the graph and over HTTP), the SSE stream and the evaluation summary. GitHub Actions runs ruff and pytest on every push.

### Project structure

```
maze_agent/
  maze.py         maze generation, loops, path search
  world.py        hazards and their rules
  levels.py       level definitions and hazard placement
  perception.py   the observation text
  brain.py        prompts, Decision schema, LLM wrapper
  models.py       model catalog and prices
  agent.py        the LangGraph graph
  runner.py       runs a level and turns the stream into events
  server.py       FastAPI application
  eval.py         evaluation harness
  __main__.py     terminal runner
static/           web page: maze canvas, graph, panels, evaluation tab
tests/            pytest suite
evals/            evaluation report
```

### Configuration

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Enables the OpenAI models |
| `ANTHROPIC_API_KEY` | Enables Claude Haiku |
| `MODEL` | Default model id (`openai:gpt-5.4-mini`) |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` | Optional LangSmith tracing |

---

## Français

Un projet d'expérimentation de **LangGraph 1.2**. Un chat piloté par un LLM doit s'échapper de six labyrinthes de difficulté croissante. Le jeu sert à mettre en œuvre les principales fonctionnalités de LangGraph dans une base de code courte et lisible : état du graphe et reducers, routage par commandes, checkpoints, interruptions avec intervention humaine, mémoire long terme, streaming, contexte d'exécution, et timeouts et relances au niveau des nœuds.

### Stack

| Paquet | Version |
|---|---|
| Python | 3.12 |
| `langgraph` | 1.2.24 |
| `langgraph-checkpoint` | 4.3.1 |
| `langchain` / `langchain-core` | 1.5.2 / 1.8.4 |
| `langchain-openai` / `langchain-anthropic` | 1.8.5 / 1.7.12 |
| `fastapi` | 0.165.0 |
| `pydantic` | 2.14.2 |

Les dépendances sont gérées avec [uv](https://docs.astral.sh/uv/). Le frontend est en HTML et JavaScript sans framework, avec Mermaid chargé depuis un CDN.

### Démarrage

```bash
cp .env.example .env              # renseigner OPENAI_API_KEY, et éventuellement ANTHROPIC_API_KEY
uv run uvicorn maze_agent.server:app
```

Ouvrez http://127.0.0.1:8000, choisissez un modèle et appuyez sur **Play**.

| Commande | Rôle |
|---|---|
| `uv run python -m maze_agent --level 5 --attempts 3` | Jouer un niveau dans le terminal, avec nouvelles tentatives après un échec |
| `uv run python -m maze_agent.eval --models <ids> --levels 1 2 3` | Lancer le banc d'évaluation |
| `uv run pytest` | Lancer les tests (sans clé API) |

### Le graphe

```
START ──> think ──> act ──┬──> think        le chat continue d'explorer
                          ├──> ask_human ──> think    le chat est bloqué : attendre le joueur
                          ├──> reflect ──> END        le chat a échoué : écrire une leçon
                          └──> END                    le chat est sorti
```

| Nœud | Type | Rôle |
|---|---|---|
| `think` | async, LLM | Construit l'observation du chat et demande au modèle une `Decision` structurée (une pensée, puis une direction ou `wait`). |
| `act` | Python pur | Applique le déplacement et les règles du monde, met à jour les vies et le statut, puis route avec `Command(goto=...)`. |
| `ask_human` | interruption | Met le graphe en pause avec `interrupt()` jusqu'à ce que le joueur envoie un indice. |
| `reflect` | async, LLM | Après un échec, écrit une leçon dans le `Store` long terme. |

#### Déroulement d'une partie

1. **Préparation.** Le niveau est généré à partir de sa graine, et l'état initial (position, monde, vies, cases visitées) est créé avec un nouveau `thread_id`.
2. **`think`.** L'observation est construite à partir de l'état (voir *Ce que perçoit le chat*). Jusqu'à trois leçons tirées des échecs précédents sur le même niveau et le même modèle sont lues dans le `Store` et ajoutées, ainsi que l'indice que le joueur vient éventuellement de donner. Le modèle répond par une `Decision` ; sa consommation de tokens et son coût sont ajoutés à l'état par des reducers.
3. **`act`.** Le déplacement est appliqué : un mur ou la porte verrouillée laissent le chat sur place, la clé est ramassée, les pics et les chiens coûtent une vie. Les chiens jouent ensuite leur tour. Le statut devient `won`, `dead`, `out_of_steps` ou reste `running`.
4. **Routage.** `act` renvoie une `Command` qui envoie le graphe vers :
   - `END` si le chat est sorti ;
   - `reflect` s'il est mort ou a épuisé ses pas ;
   - `ask_human` s'il est bloqué (voir ci-dessous) et qu'un joueur est connecté ;
   - `think` sinon.
5. **`ask_human`.** Le chat est considéré comme bloqué quand sa carte mentale n'a plus d'ouverture inexplorée, ou quand ses 8 dernières positions couvrent 2 cases ou moins. Un délai de 15 pas évite les appels répétés. Le graphe s'arrête avec `interrupt()` ; le serveur web clôt le flux, et la réponse du joueur reprend le même fil avec `Command(resume=...)`. L'indice est montré au LLM au `think` suivant.
6. **`reflect`.** Le modèle reçoit un résumé de la tentative (résultat, cases explorées, état de la clé, chaque blessure avec son pas et sa position) et écrit une leçon de 30 mots au plus, rangée sous l'espace de noms `("lessons", "<niveau>:<modèle>")`.

### Le jeu

#### Niveaux

| # | Nom | Taille | Pics | Chiens | Porte | Passages en plus | Vies |
|---|---|---|---|---|---|---|---|
| 1 | First steps | 7×7 | | | | | 3 |
| 2 | Getting lost | 10×10 | | | | | 3 |
| 3 | Spikes | 12×12 | 8 | | | 3 | 3 |
| 4 | Dogs | 14×14 | 8 | 3 | | 8 | 3 |
| 5 | Locked door | 16×16 | 10 | 3 | ✓ | 10 | 3 |
| 6 | The final maze | 18×18 | 14 | 4 | ✓ | 14 | 5 |

Chaque niveau autorise `2 × taille²` pas.

#### Génération

- **Labyrinthe.** Un algorithme « growing tree » à graine fixe creuse un labyrinthe parfait, en partant de la dernière case une fois sur deux et d'une case au hasard sinon. On obtient beaucoup de carrefours et des culs-de-sac courts. Quelques murs sont ensuite ouverts pour créer des boucles, afin de pouvoir contourner un danger.
- **Porte et clé.** La porte se trouve aux trois quarts du chemin de sortie. La clé est placée avant la porte, sur la case la plus éloignée à la fois du départ et de la porte. Les boucles ne relient jamais les deux côtés de la porte, qu'on ne peut donc pas contourner.
- **Pics.** Placés hors du chemin de sortie, aussi près de lui que possible (entrées des couloirs latéraux d'abord), jamais derrière un autre pic ni sur le chemin de la clé.
- **Chiens.** Placés hors du chemin, assez loin du départ, sur des cases d'où ils peuvent rejoindre le chemin de sortie sans traverser de pics.
- Chaque niveau peut être réussi sans perdre de vie ; les tests le vérifient pour les six.

#### Règles

- À chaque tour, le chat avance d'une case ou attend.
- Les pics coûtent une vie à chaque passage.
- Les chiens se déplacent un tour sur deux. Ils patrouillent en avançant et ne font demi-tour qu'au bout d'un cul-de-sac ; ils ne poursuivent jamais le chat. Ils évitent les pics, la porte verrouillée et le départ. Un chien qui mord retourne à sa case de départ.
- La porte bloque tout le monde tant que le chat n'a pas la clé.

#### Ce que perçoit le chat

L'observation envoyée au LLM contient :

- sa position, la sortie, le numéro du pas, ses vies et s'il porte la clé ;
- ce qui s'est passé au tour précédent (mur, porte verrouillée, pics, morsure, clé) ;
- chaque direction ouverte avec la case voisine, le nombre de visites et ce qui s'y trouve (sortie, chien, pics, clé, porte), plus un avertissement quand un chien se tient juste à côté de cette case ;
- les chiens qu'il entend (à 3 cases ou moins) et, pour chacun, s'il se rapproche ou s'éloigne ;
- une indication de **carte mentale** : l'ouverture inexplorée la plus proche, accessible par des cases déjà visitées, en évitant les pics et la porte verrouillée ;
- les leçons des tentatives précédentes et l'indice du joueur, s'il y en a.

### Fonctionnalités LangGraph utilisées

| Fonctionnalité | Où | Utilité |
|---|---|---|
| `StateGraph` avec un état Pydantic | `agent.py` `State` | L'état reste typé après rechargement d'un checkpoint. Un simple dict transformerait les tuples en listes. |
| Reducers (`Annotated[..., add]`) | `State.tokens`, `cost`, `incidents` | Les nœuds ne renvoient que leur part ; LangGraph additionne tokens et coût et ajoute les blessures. |
| Contexte d'exécution (`context_schema`, `Runtime[Context]`) | `Context` | Injecte le labyrinthe, le cerveau LLM et les options de la partie. Les tests injectent un cerveau scripté. |
| Nœuds async avec `timeout` et `RetryPolicy` | `think`, `reflect` | Un appel LLM bloqué est annulé après 30 s et relancé jusqu'à 3 fois. |
| `Command(update=..., goto=...)` et `destinations` | `act` | Un même nœud met à jour l'état et choisit le nœud suivant ; `destinations` déclare les cibles possibles pour le diagramme. |
| Checkpointer (`InMemorySaver`) | compilation du graphe | Sauvegarde chaque étape ; indispensable aux interruptions. Une liste blanche `JsonPlusSerializer` déclare les types maison qu'il peut restaurer. |
| `interrupt()` et `Command(resume=...)` | `ask_human`, `server.py` | Met une partie en pause et reprend le même fil avec la réponse du joueur. |
| Mémoire long terme (`InMemoryStore`, `runtime.store`) | `reflect`, `think` | Les leçons persistent d'une partie à l'autre, avec un espace de noms par niveau et par modèle. |
| Streaming (`astream`, `version="v2"`, `updates` + `messages`) | `runner.py` | `updates` anime le plateau ; `messages` transporte les tokens du LLM de `think`, filtrés par `langgraph_node`. |
| Introspection du graphe (`get_graph().draw_mermaid()`) | `/api/graph` | Le graphe compilé est affiché dans l'interface, et le nœud en cours est mis en évidence. |
| Configuration d'exécution (`tags`, `metadata`, `recursion_limit`) | `runner.py` | Étiquette chaque partie avec son niveau et son modèle, pour LangSmith quand le tracing est activé. |

Côté LangChain, `init_chat_model` construit le modèle de n'importe quel fournisseur à partir d'un seul identifiant, et `with_structured_output(method="json_schema")` contraint la réponse au schéma `Decision` grâce à la sortie structurée native du fournisseur.

### Application web

Le serveur FastAPI diffuse les parties vers le navigateur en Server-Sent Events natifs (`fastapi.sse`).

| Endpoint | Description |
|---|---|
| `GET /api/levels` | Niveaux et modèles disponibles |
| `GET /api/levels/{n}` | Labyrinthe et dangers d'un niveau |
| `GET /api/levels/{n}/play?model=` | Lance une partie (SSE) |
| `GET /api/runs/{thread_id}/resume?answer=` | Reprend une partie en pause avec l'indice du joueur (SSE) |
| `GET /api/lessons?level=&model=` / `DELETE` | Lit ou efface les leçons enregistrées |
| `GET /api/graph` | Source Mermaid du graphe compilé |
| `GET /api/eval` | Dernier rapport d'évaluation |

Une partie émet les événements `start`, `token`, `think`, `act`, `interrupt`, `ask_human`, `reflect` et `done`. Fermer la page ferme le flux, ce qui annule le graphe et arrête les appels au LLM.

La page affiche le labyrinthe, le graphe en direct, les pensées du chat au fil de leur génération (chacune dépliable jusqu'au prompt exact), les leçons enregistrées, un panneau d'aide quand le chat demande un indice, et un onglet **Evaluation**.

### Modèles

| Modèle | Prix entrée / sortie par million de tokens |
|---|---|
| `openai:gpt-5.4-mini` (par défaut) | 0,75 $ / 4,50 $ |
| `openai:gpt-5.4-nano` | 0,20 $ / 1,25 $ |
| `anthropic:claude-haiku-5-5` | 0,10 $ / 0,50 $ |

Un modèle n'est sélectionnable que si sa clé API est renseignée ; sinon il apparaît grisé dans le sélecteur. La consommation de tokens provient de chaque réponse, et le coût d'une partie est calculé à partir de ce tableau.

### Évaluation

`maze_agent/eval.py` joue des chaînes de tentatives pour chaque modèle et chaque niveau. Chaque chaîne s'arrête à sa première victoire, et chaque tentative lit les leçons écrites par les précédentes. Pour chaque couple `(modèle, niveau)`, le rapport indique :

- le taux de réussite à la première tentative, et dans la chaîne ;
- le nombre moyen de pas des tentatives gagnantes, comparé au chemin optimal (clé comprise quand il y a une porte) ;
- le nombre moyen de vies perdues ;
- le coût moyen par tentative, et le coût total par victoire.

Le rapport est écrit dans `evals/results.json` et affiché dans l'onglet Evaluation. `--update` remesure les modèles et niveaux demandés et conserve le reste du rapport.

### Tests

Les tests utilisent un cerveau scripté et tournent donc sans clé API. Ils couvrent la génération des labyrinthes, les contraintes des niveaux, les règles du monde, la carte mentale, le texte de l'observation, le calcul des coûts, le cycle des leçons dans le `Store`, le cycle interruption et reprise (dans le graphe et en HTTP), le flux SSE et le résumé d'évaluation. GitHub Actions lance ruff et pytest à chaque push.

### Structure du projet

```
maze_agent/
  maze.py         génération des labyrinthes, boucles, recherche de chemin
  world.py        dangers et leurs règles
  levels.py       définition des niveaux et placement des dangers
  perception.py   le texte de l'observation
  brain.py        prompts, schéma Decision, enveloppe du LLM
  models.py       catalogue des modèles et prix
  agent.py        le graphe LangGraph
  runner.py       joue un niveau et transforme le flux en événements
  server.py       application FastAPI
  eval.py         banc d'évaluation
  __main__.py     exécution en terminal
static/           page web : canvas du labyrinthe, graphe, panneaux, onglet d'évaluation
tests/            suite pytest
evals/            rapport d'évaluation
```

### Configuration

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Active les modèles OpenAI |
| `ANTHROPIC_API_KEY` | Active Claude Haiku |
| `MODEL` | Modèle par défaut (`openai:gpt-5.4-mini`) |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` | Tracing LangSmith (optionnel) |
