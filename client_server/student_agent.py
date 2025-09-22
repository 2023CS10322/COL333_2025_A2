"""
StudentAgent — River & Stones (Minimax)
- Classic minimax + alpha–beta (no negamax), iterative deepening (MAX_DEPTH=4)
- Score-first evaluation: n (scored stones) >> m (1-move-to-score)
- Threat-aware: block if opponent can score in 1
- Branching under control (flip/rotate budget = 1), strict time/node guards
"""

import time, copy, random
from typing import List, Dict, Any, Optional, Tuple

# Framework helpers (from agent.py)
from agent import (
    BaseAgent,
    agent_compute_valid_moves,
    agent_apply_move,
    is_own_score_cell,
    get_opponent,
)

INF = 10**9


class StudentAgent(BaseAgent):
    # ----- Evaluation Weights (make scoring dwarf everything else) -----
    W_N = 10000   # stones already scored (ours - theirs)
    W_M = 2000    # stones that can score in one move (ours - theirs)
    W_MOB = 5     # small tie-breaker
    W_POS = 1     # tiny positional tie-break

    # ----- Search Limits -----
    PER_MOVE_TIME = 0.25
    MAX_NODES = 6000
    MAX_DEPTH = 8

    def __init__(self, player: str):
        super().__init__(player)
        self.rng = random.Random(1337)

    # ====================== Public API ======================
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> Optional[Dict[str, Any]]:
        """Return a legal move dict for self.player, or None if no legal moves."""
        root_moves = self._generate_all_moves(board, self.player, rows, cols, score_cols)
        if not root_moves:
            return None

        # (1) Immediate scorer: if any move increases n, play it immediately
        n0 = self._count_n(board, self.player, rows, cols, score_cols)
        for mv in root_moves:
            ok, nb = self._sim_apply(board, mv, self.player, rows, cols, score_cols)
            if ok and self._count_n(nb, self.player, rows, cols, score_cols) > n0:
                return mv

        # (2) If opponent can score in 1, restrict to defensive candidates
        if self._count_m(board, get_opponent(self.player), rows, cols, score_cols) > 0:
            defensive = self._defensive_candidates(board, root_moves, rows, cols, score_cols)
            if defensive:
                root_moves = defensive

        # (3) Iterative deepening minimax (root is maximizing for self.player)
        ordered = self._order_moves(board, root_moves, rows, cols, score_cols, self.player)
        deadline = time.time() + self.PER_MOVE_TIME
        best_move, best_val = ordered[0], -INF
        nodes_total = 0

        depth = 1
        while depth <= self.MAX_DEPTH:
            try:
                alpha, beta = -INF, INF
                cur_best, cur_val = None, -INF

                for mv in ordered:
                    if time.time() >= deadline or nodes_total >= self.MAX_NODES:
                        raise TimeoutError
                    ok, nb = self._sim_apply(board, mv, self.player, rows, cols, score_cols)
                    if not ok:
                        continue
                    val, used = self._search(nb, depth - 1, alpha, beta, get_opponent(self.player), deadline)
                    nodes_total += used

                    if val > cur_val:
                        cur_val, cur_best = val, mv
                        if cur_val > alpha:
                            alpha = cur_val

                if cur_best is not None:
                    best_move, best_val = cur_best, cur_val

                depth += 1
            except TimeoutError:
                break

        return best_move

    # ====================== Minimax + Alpha–Beta ======================
    def _search(self, board, depth: int, alpha: int, beta: int,
                player_to_move: str, deadline: float) -> Tuple[int, int]:
        """
        Minimax with alpha–beta pruning.
        Returns (score_from_self_perspective, nodes_used).
        """
        nodes_used = 0

        if time.time() >= deadline:
            raise TimeoutError

        # Quick terminal via n>=4 (engine’s win condition)
        n_self = self._count_n(board, self.player, rows=len(board), cols=len(board[0]), score_cols=self._score_cols_from_board(board))
        n_opp  = self._count_n(board, get_opponent(self.player), rows=len(board), cols=len(board[0]), score_cols=self._score_cols_from_board(board))
        if n_self >= 4:
            return 100_000, 0
        if n_opp >= 4:
            return -100_000, 0

        if depth <= 0:
            return self._eval(board, len(board), len(board[0]), self._score_cols_from_board(board)), 1

        rows = len(board); cols = len(board[0]); score_cols = self._score_cols_from_board(board)
        moves = self._generate_all_moves(board, player_to_move, rows, cols, score_cols)
        if not moves:
            return self._eval(board, rows, cols, score_cols), 1

        moves = self._order_moves(board, moves, rows, cols, score_cols, player_to_move)

        maximizing = (player_to_move == self.player)

        if maximizing:
            best = -INF
            for mv in moves:
                if time.time() >= deadline:
                    raise TimeoutError
                ok, nb = self._sim_apply(board, mv, player_to_move, rows, cols, score_cols)
                if not ok:
                    continue
                val, used = self._search(nb, depth - 1, alpha, beta, get_opponent(player_to_move), deadline)
                nodes_used += (1 + used)
                if val > best:
                    best = val
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
                if nodes_used >= self.MAX_NODES:
                    break
            return best, nodes_used
        else:
            best = +INF
            for mv in moves:
                if time.time() >= deadline:
                    raise TimeoutError
                ok, nb = self._sim_apply(board, mv, player_to_move, rows, cols, score_cols)
                if not ok:
                    continue
                val, used = self._search(nb, depth - 1, alpha, beta, get_opponent(player_to_move), deadline)
                nodes_used += (1 + used)
                if val < best:
                    best = val
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break
                if nodes_used >= self.MAX_NODES:
                    break
            return best, nodes_used

    # ====================== Evaluation ======================
    def _eval(self, board, rows, cols, score_cols) -> int:
        """Score-first heuristic: n >> m; tiny mobility/positional tie-breakers."""
        us = self.player
        them = get_opponent(us)

        n_us   = self._count_n(board, us, rows, cols, score_cols)
        n_them = self._count_n(board, them, rows, cols, score_cols)
        m_us   = self._count_m(board, us, rows, cols, score_cols)
        m_them = self._count_m(board, them, rows, cols, score_cols)

        mob_us   = self._mobility(board, us, rows, cols, score_cols)
        mob_them = self._mobility(board, them, rows, cols, score_cols)

        pos = self._positional_tiebreak(board, rows, cols, score_cols)

        return (self.W_N * (n_us - n_them) +
                self.W_M * (m_us - m_them) +
                self.W_MOB * (mob_us - mob_them) +
                self.W_POS * pos)

    def _count_n(self, board, player, rows, cols, score_cols) -> int:
        total = 0
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if p and p.owner == player and getattr(p, "side", "stone") == "stone":
                    if is_own_score_cell(x, y, player, rows, cols, score_cols):
                        total += 1
        return total

    def _count_m(self, board, player, rows, cols, score_cols) -> int:
        """
        Stones of 'player' that can reach their scoring cell in ONE legal move
        (including river sweeps and pushes). Count each stone once.
        """
        m = 0
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if not p or p.owner != player or getattr(p, "side", "stone") != "stone":
                    continue
                info = agent_compute_valid_moves(board, x, y, player, rows, cols, score_cols)
                # direct/river moves
                for (tx, ty) in info.get("moves", ()):
                    if is_own_score_cell(tx, ty, player, rows, cols, score_cols):
                        m += 1
                        break
                else:
                    # pushes: own_final (ox,oy) is where OUR piece goes
                    for ((ox, oy), (_px, _py)) in info.get("pushes", ()):
                        if is_own_score_cell(ox, oy, player, rows, cols, score_cols):
                            m += 1
                            break
        return m

    def _mobility(self, board, player, rows, cols, score_cols) -> int:
        total = 0
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if not p or p.owner != player:
                    continue
                info = agent_compute_valid_moves(board, x, y, player, rows, cols, score_cols)
                total += len(info.get("moves", ())) + len(info.get("pushes", ()))
        return total

    def _positional_tiebreak(self, board, rows, cols, score_cols) -> int:
        """Reward being closer to own scoring row (small effect)."""
        def own_row(plr: str) -> int:
            # Find the y that contains any own scoring cell for plr
            for yy in range(rows):
                for xx in score_cols:
                    if is_own_score_cell(xx, yy, plr, rows, cols, score_cols):
                        return yy
            return 0 if plr == "circle" else rows - 1

        def score_player(plr: str) -> int:
            s = 0
            tr = own_row(plr)
            for y in range(rows):
                for x in range(cols):
                    p = board[y][x]
                    if p and p.owner == plr and getattr(p, "side", "stone") == "stone":
                        s -= abs(y - tr)
            return s

        return score_player(self.player) - score_player(get_opponent(self.player))

    # ====================== Move Gen & Ordering ======================
    def _generate_all_moves(self, board, player, rows, cols, score_cols) -> List[Dict[str, Any]]:
        """
        Build all legal move dicts for 'player'.
        - Moves & pushes via agent_compute_valid_moves (rule-correct).
        - Include a tiny budget of flip/rotate options (engine validates).
        """
        out: List[Dict[str, Any]] = []
        flip_rotate_budget = 1   # SMALL to keep branching down
        flips_added = 0

        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if not p or p.owner != player:
                    continue

                info = agent_compute_valid_moves(board, x, y, player, rows, cols, score_cols)

                for (tx, ty) in info.get("moves", ()):
                    out.append({"action": "move", "from": [x, y], "to": [tx, ty]})

                for ((ox, oy), (px, py)) in info.get("pushes", ()):
                    out.append({"action": "push",
                                "from": [x, y],
                                "to": [ox, oy],
                                "pushed_to": [px, py]})

                # budgeted flips/rotates (one piece contributes at most once)
                if flips_added < flip_rotate_budget:
                    if getattr(p, "side", "stone") == "stone":
                        out.append({"action": "flip", "from": [x, y], "orientation": "horizontal"})
                        out.append({"action": "flip", "from": [x, y], "orientation": "vertical"})
                    else:
                        out.append({"action": "flip", "from": [x, y]})
                        out.append({"action": "rotate", "from": [x, y]})
                    flips_added += 1
                    # don't add more flip/rotates for this same piece in this ply
                    continue

        return out

    def _order_moves(self, board, moves, rows, cols, score_cols, player_to_move) -> List[Dict[str, Any]]:
        """
        Ordering for strong αβ pruning:
        1) finishers (make n>=4),
        2) urgent defense (reduce opponent m),
        3) pushes,
        4) flips/rotates,
        5) other moves (mild centrality).
        """
        current_n = self._count_n(board, player_to_move, rows, cols, score_cols)
        opp = get_opponent(player_to_move)
        m_opp0 = self._count_m(board, opp, rows, cols, score_cols)

        def score_move(mv: Dict[str, Any]) -> int:
            s = 0
            # Finisher?
            if self._is_finisher_after(board, mv, current_n, rows, cols, score_cols, player_to_move):
                s += 10000

            # Urgent defense: reduce opponent 1-move scoring chances
            ok, nb = self._sim_apply(board, mv, player_to_move, rows, cols, score_cols)
            if ok:
                m_opp1 = self._count_m(nb, opp, rows, cols, score_cols)
                if m_opp1 < m_opp0:
                    s += 1500

            # Tactical > structural
            if mv.get("action") == "push":
                s += 300
            elif mv.get("action") in ("flip", "rotate"):
                s += 60
            elif mv.get("action") == "move":
                fr = mv.get("from", [0, 0]); to = mv.get("to", fr)
                s += -(abs(to[0] - cols // 2) + abs(to[1] - rows // 2))

            return s

        return sorted(moves, key=score_move, reverse=True)

    # ====================== Helpers ======================
    def _sim_apply(self, board, move, player, rows, cols, score_cols):
        """Apply a move on a deep-copied board using the engine validator."""
        bcopy = copy.deepcopy(board)
        ok, msg = agent_apply_move(bcopy, move, player, rows, cols, score_cols)
        return ok, (bcopy if ok else msg)

    def _is_finisher_after(self, board, move, current_n, rows, cols, score_cols, mover):
        ok, nb = self._sim_apply(board, move, mover, rows, cols, score_cols)
        return ok and self._count_n(nb, mover, rows, cols, score_cols) >= 4

    def _defensive_candidates(self, board, moves, rows, cols, score_cols):
        """Keep moves that reduce opponent m (1-move to score)."""
        them = get_opponent(self.player)
        m0 = self._count_m(board, them, rows, cols, score_cols)
        keep = []
        for mv in moves:
            ok, nb = self._sim_apply(board, mv, self.player, rows, cols, score_cols)
            if ok and self._count_m(nb, them, rows, cols, score_cols) < m0:
                keep.append(mv)
        return keep

    def _score_cols_from_board(self, board):
        """Infer score_cols from board width: middle 4 columns as used by engine helpers."""
        cols = len(board[0])
        w = 4
        start = max(0, (cols - w) // 2)
        return list(range(start, start + w))
