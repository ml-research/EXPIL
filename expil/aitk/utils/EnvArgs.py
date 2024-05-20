# Created by jing at 09.02.24
import torch
import shutil
from collections import deque


class EnvArgs():
    """ generate one micro-program
    """

    def __init__(self, agent, args, window_size, fps):
        super().__init__()
        # game setting
        self.jump_frames = args.jump_frames
        self.device = args.device

        # self.output_folder = args.game_buffer_path
        # self.max_lives = args.max_lives
        # self.reward_lost_one_live = args.reward_lost_one_live
        # layout setting
        args.zoom_in = 2.5
        args.stack_num = 10
        self.zoom_in = args.zoom_in
        self.db_num = 4
        if window_size is not None:
            self.width_game_window = int(window_size[1] * args.zoom_in)
            self.height_game_window = int(window_size[0] * args.zoom_in)
            self.width_left_panel = int(window_size[0] * args.zoom_in)
            self.width_right_panel = int(window_size[1] * 0.25 * args.zoom_in)
            self.position_norm_factor = window_size[0]
            self.last_obs = torch.zeros((window_size[0], window_size[1], 3), dtype=torch.uint8).numpy()
        # frame rate limiting
        self.fps = fps
        self.target_frame_duration = 1 / fps
        self.last_frame_time = 0
        # record and statistical properties

        self.past_states = deque(maxlen=args.stack_num)
        self.past_actions = deque(maxlen=args.stack_num)

        self.action = None
        self.collective = None
        self.target = None
        self.logic_state = None
        self.last_state = None
        self.last2nd_state = None
        self.next_state = None
        self.reward = None
        self.obs = None
        self.game_next_states = []
        self.game_states = []
        self.game_actions = []
        self.game_rewards = []
        self.game_obj_types = []
        self.game_relations = []
        self.rule_data_buffer = []
        self.game_i = 0
        self.win_count = 0
        self.dead_counter = 0
        self.current_steak = 0
        self.explain_text = ""
        self.game_num = args.teacher_game_nums
        self.win_rate = torch.zeros(self.game_num)
        self.learn_performance = torch.zeros(self.game_num)

    def reset_args(self, game_i):
        self.game_i = game_i
        self.frame_i = 0
        self.current_lives = self.max_lives
        self.state_score = 0
        self.state_loss = 0
        self.game_over = False
        self.terminated = False
        self.truncated = False

    def update_args(self):
        self.frame_i += 1
        if self.state_score > self.best_score:
            self.best_score = self.state_score

        if self.reward < 0:
            self.score_update = True
            self.current_steak = 0
        elif self.reward > 0:
            self.current_steak += 1
            self.max_steak = max(self.max_steak, self.current_steak)
            if self.max_steak >= self.mile_stone_scores[0] and not self.has_win_2:
                self.has_win_2 = True
                self.win_2 = self.game_i
            if self.max_steak >= self.mile_stone_scores[1] and not self.has_win_3:
                self.has_win_3 = True
                self.win_3 = self.game_i
            if self.max_steak >= self.mile_stone_scores[2] and not self.has_win_5:
                self.has_win_5 = True
                self.win_5 = self.game_i
            self.score_update = True
        else:
            self.score_update = False

    def update_lost_live(self, game_name, current_live):
        self.current_lives = current_live
        self.score_update = True
        if game_name == "Kangaroo":
            self.reward = self.reward_lost_one_live
        #     self.rewards[-1] += self.reward_lost_one_live
        #     self.dead_counter += 1
        # if game_name == "Asterix":
        #     self.reward = self.reward_lost_one_live
        #     self.rewards[-1] += self.reward_lost_one_live
        #     self.dead_counter += 1

    def buffer_frame(self, buffer_type):
        self.next_states.append(self.next_state)
        self.logic_states.append(self.logic_state)
        self.rewards.append(self.reward)
        if buffer_type == "dqn_a":
            self.actions.append(self.action)
        elif buffer_type == "dqn_c":
            self.actions.append(self.collective)
        elif buffer_type == "dqn_t":
            self.actions.append(self.target)
        else:
            raise ValueError

    def buffer_game(self, zero_reward, save_frame):
        states = []
        actions = []
        rewards = []
        next_states = []
        for f_i, reward in enumerate(self.rewards):
            next_states.append(self.next_states[f_i])
            states.append(self.logic_states[f_i])
            actions.append(self.actions[f_i])
            rewards.append(self.rewards[f_i])
            if save_frame:
                # move dead frame to some folder
                shutil.copy2(self.output_folder / "frames" / f"g_{self.game_i}_f_{f_i}.png",
                             self.output_folder / "key_frames" / f"g_{self.game_i}_f_{f_i}.png")
        self.game_states.append(states)
        self.game_rewards.append(rewards)
        self.game_actions.append(actions)
        self.game_next_states.append(next_states)

    def reset_buffer_game(self):
        self.next_states = []
        self.logic_states = []
        self.actions = []
        self.rewards = []
