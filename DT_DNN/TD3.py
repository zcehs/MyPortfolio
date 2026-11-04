import tensorflow.compat.v1 as tf
import numpy as np


# 使用 TensorFlow 1.x graph/session 模式
tf.disable_v2_behavior()


class TD3(object):
    def __init__(self, a_dim, s_dim, a_bound, name, sess, memory_capacity, gamma, lr_a, lr_c, tau, batch_size):
        self.name = name
        self.sess = sess

        self.a_dim = int(a_dim)
        self.s_dim = int(s_dim)

        # 支援 scalar 或 list/array 類型的 action bound
        self.a_bound = np.asarray(a_bound, dtype=np.float32)

        if self.a_bound.ndim == 0:
            self.a_bound = np.full(self.a_dim, float(self.a_bound), dtype=np.float32)

        self.memory_capacity = int(memory_capacity)

        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.tau = tau
        self.batch_size = batch_size

        self.pointer = 0
        self.update_cnt = 0

        # TD3 通常延遲更新 Actor
        self.policy_target_update_interval = 2

        # Target policy smoothing
        self.policy_noise = 0.2
        self.noise_clip = 0.5

        # Replay buffer:
        # state + action + reward + next_state
        self.memory = np.zeros((self.memory_capacity, self.s_dim * 2 + self.a_dim + 1), dtype=np.float32)

        # --------------------------------------------------
        # Placeholders
        # --------------------------------------------------

        self.S = tf.placeholder(tf.float32, shape=[None, self.s_dim], name=f"{name}_s")
        self.S_ = tf.placeholder(tf.float32, shape=[None, self.s_dim], name=f"{name}_s_next")
        self.A = tf.placeholder(tf.float32, shape=[None, self.a_dim], name=f"{name}_action")
        self.R = tf.placeholder( tf.float32, shape=[None, 1], name=f"{name}_reward")
        actor_scope = f"{name}/Actor"
        critic_scope = f"{name}/Critic"

        # --------------------------------------------------
        # Actor networks
        # --------------------------------------------------

        with tf.variable_scope(actor_scope):
            self.a = self._build_a(self.S, scope="eval", trainable=True)
            self.a_target = self._build_a(self.S_, scope="target", trainable=False)

        # Target policy smoothing noise
        target_noise = tf.random_normal(shape=tf.shape(self.a_target), mean=0.0, stddev=self.policy_noise, dtype=tf.float32)
        target_noise = tf.clip_by_value( target_noise, -self.noise_clip, self.noise_clip)
        action_bound_tensor = tf.constant(self.a_bound, dtype=tf.float32)
        noisy_target_action = tf.clip_by_value( self.a_target + target_noise, 0.0, action_bound_tensor)

        # --------------------------------------------------
        # Critic networks
        # --------------------------------------------------

        with tf.variable_scope(critic_scope):
            # Critic 1：replay buffer action
            self.q1 = self._build_c(self.S, self.A, scope="eval1", trainable=True, reuse=False)
            # Critic 2：replay buffer action
            self.q2 = self._build_c(self.S, self.A, scope="eval2", trainable=True, reuse=False)
            # Actor 更新時使用 Critic 1 評估 actor action
            self.q1_actor = self._build_c(self.S, self.a, scope="eval1", trainable=True, reuse=True)
            # Target critics
            self.q1_target = self._build_c(self.S_, noisy_target_action, scope="target1", trainable=False, reuse=False)
            self.q2_target = self._build_c(self.S_, noisy_target_action, scope="target2", trainable=False, reuse=False)

        # --------------------------------------------------
        # Collect parameters
        # --------------------------------------------------

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f"{actor_scope}/eval")
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f"{actor_scope}/target")
        self.ce_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f"{critic_scope}/eval1")
        self.ct_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f"{critic_scope}/target1")
        self.ce_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f"{critic_scope}/eval2")
        self.ct_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f"{critic_scope}/target2")

        # --------------------------------------------------
        # TD target
        # --------------------------------------------------

        min_target_q = tf.minimum(self.q1_target, self.q2_target)
        q_target = self.R + self.gamma * min_target_q
        # Target 不參與反向傳播
        q_target = tf.stop_gradient(q_target)

        # --------------------------------------------------
        # Critic losses
        # --------------------------------------------------

        self.c_loss1 = tf.reduce_mean(tf.square(q_target - self.q1))
        self.c_loss2 = tf.reduce_mean(tf.square(q_target - self.q2))
        self.ctrain1 = tf.train.AdamOptimizer(self.lr_c).minimize(self.c_loss1, var_list=self.ce_params1)
        self.ctrain2 = tf.train.AdamOptimizer(self.lr_c).minimize(self.c_loss2, var_list=self.ce_params2)

        # --------------------------------------------------
        # Actor loss
        # 最大化 Q，所以最小化 -Q
        # --------------------------------------------------

        self.a_loss = -tf.reduce_mean(self.q1_actor)
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(self.a_loss, var_list=self.ae_params)

        # --------------------------------------------------
        # Target network update
        # --------------------------------------------------

        self.hard_replace = []
        for target_param, eval_param in zip(self.at_params, self.ae_params):
            self.hard_replace.append(tf.assign(target_param, eval_param))

        for target_param, eval_param in zip(self.ct_params1, self.ce_params1):
            self.hard_replace.append(tf.assign(target_param, eval_param))

        for target_param, eval_param in zip(self.ct_params2, self.ce_params2):
            self.hard_replace.append(tf.assign(target_param, eval_param))

        self.soft_replace = []
        for target_param, eval_param in zip(self.at_params, self.ae_params):
            self.soft_replace.append(tf.assign(target_param, (1.0 - self.tau) * target_param + self.tau * eval_param))

        for target_param, eval_param in zip(self.ct_params1,self.ce_params1):
            self.soft_replace.append(tf.assign(target_param, (1.0 - self.tau) * target_param + self.tau * eval_param))

        for target_param, eval_param in zip(self.ct_params2, self.ce_params2):
            self.soft_replace.append(tf.assign(target_param, (1.0 - self.tau) * target_param + self.tau * eval_param))

    def initialize_target_networks(self):
        """
        必須在 tf.global_variables_initializer() 執行後呼叫。
        將 eval network 的初始參數複製到 target network。
        """
        self.sess.run(self.hard_replace)

    def choose_action(self, s):
        s = np.asarray(s, dtype=np.float32)

        if s.ndim == 1:
            s = np.expand_dims(s, axis=0)

        if s.shape[1] != self.s_dim:
            raise ValueError(f"{self.name} state 維度錯誤："f"收到 {s.shape[1]}，預期 {self.s_dim}")

        return self.sess.run(self.a,feed_dict={self.S: s})

    def store_transition(self, s, a, r, s_):
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        s_ = np.asarray(s_, dtype=np.float32)

        if s.ndim == 1:
            s = np.expand_dims(s, axis=0)
        if a.ndim == 1:
            a = np.expand_dims(a, axis=0)
        if s_.ndim == 1:
            s_ = np.expand_dims(s_, axis=0)
        if s.shape[1] != self.s_dim:raise ValueError(f"{self.name} current state 維度錯誤："f"{s.shape[1]}，預期 {self.s_dim}")
        if s_.shape[1] != self.s_dim:
            raise ValueError(f"{self.name} next state 維度錯誤："f"{s_.shape[1]}，預期 {self.s_dim}。"f"這通常發生在 DT/DNN 模式切換時。")
        if a.shape[1] != self.a_dim:
            raise ValueError(f"{self.name} action 維度錯誤："f"{a.shape[1]}，預期 {self.a_dim}")

        transition = np.hstack((s[0], a[0], [float(r)], s_[0]))

        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def learn(self):
        current_memory_size = min(self.pointer, self.memory_capacity)

        if current_memory_size < self.batch_size:
            return

        self.update_cnt += 1

        indices = np.random.choice(current_memory_size, size=self.batch_size, replace=False)

        batch = self.memory[indices, :]
        bs = batch[:, :self.s_dim]
        ba = batch[:, self.s_dim: self.s_dim + self.a_dim]
        br = batch[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
        bs_ = batch[:, -self.s_dim:]
        feed_dict = {self.S: bs, self.A: ba, self.R: br, self.S_: bs_}

        # 每一步更新兩個 Critic
        self.sess.run(self.ctrain1, feed_dict=feed_dict)

        self.sess.run(self.ctrain2, feed_dict=feed_dict)

        # 延遲更新 Actor 和 target networks
        if (self.update_cnt % self.policy_target_update_interval == 0):
            self.sess.run(self.atrain, feed_dict={self.S: bs})
            self.sess.run(self.soft_replace)

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 600, activation=tf.nn.relu, name="l1", trainable=trainable)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu, name="l2", trainable=trainable)
            l3 = tf.layers.dense(l2, 200, activation=tf.nn.relu, name="l3", trainable=trainable)
            l4 = tf.layers.dense(l3, 100, activation=tf.nn.relu, name="l4", trainable=trainable)
            l5 = tf.layers.dense(l4, 50, activation=tf.nn.relu, name="l5", trainable=trainable)
            normalized_action = tf.layers.dense(l5, self.a_dim, activation=tf.nn.sigmoid, name="action", trainable=trainable)
            action_bound_tensor = tf.constant(self.a_bound, dtype=tf.float32)
            return normalized_action * action_bound_tensor

    def _build_c(self, s, a, scope, trainable, reuse=False
    ):
        with tf.variable_scope(scope, reuse=reuse):
            n_l1 = 600
            w1_s = tf.get_variable("w1_s", shape=[self.s_dim, n_l1], trainable=trainable, initializer=tf.glorot_uniform_initializer())
            w1_a = tf.get_variable("w1_a", shape=[self.a_dim, n_l1], trainable=trainable, initializer=tf.glorot_uniform_initializer())
            b1 = tf.get_variable("b1", shape=[1, n_l1], trainable=trainable, initializer=tf.zeros_initializer())
            l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu, name="l2", trainable=trainable)
            l3 = tf.layers.dense(l2, 200, activation=tf.nn.relu, name="l3", trainable=trainable)
            l4 = tf.layers.dense(l3, 100, activation=tf.nn.relu, name="l4", trainable=trainable)
            l5 = tf.layers.dense(l4, 50, activation=tf.nn.relu, name="l5", trainable=trainable)
            q = tf.layers.dense(l5, 1, activation=None, name="q", trainable=trainable)
            return q