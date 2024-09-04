import tensorflow  as tf

# @title Jitter model
class Jitter_model(tf.keras.Model):
    min_max_scores_fields = {
        "flow_traffic",
        "flow_packets",
        "flow_packet_size",
        "link_capacity",
    }

    name = "Jitter_model"

    def __init__(self, override_min_max_scores=None, name=None):
        super(Jitter_model, self).__init__()

        self.iterations = 12
        # check 6?
        self.path_state_dim = 128
        self.link_state_dim = 128

        if override_min_max_scores is not None:
            self.set_min_max_scores(override_min_max_scores)
        if name is not None:
            assert type(name) == str, "name must be a string"
            self.name = name

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate"),
            return_sequences=True,
            return_state=True,
            name="PathUpdateRNN",
        )
        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate"
        )

        self.device_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="DeviceUpdate"
        )

        self.path_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3)),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="PathEmbedding",
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=3),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="LinkEmbedding",
        )

        self.device_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=2),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="DeviceEmbedding",
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=3),
                tf.keras.layers.Dense(
                    self.link_state_dim // 2, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 8, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
            ],
            name="PathReadout",
        )

    def set_min_max_scores(self, override_min_max_scores):
        assert (
            type(override_min_max_scores) == dict
            and all(kk in override_min_max_scores for kk in self.min_max_scores_fields)
            and all(len(val) == 2 for val in override_min_max_scores.values())
        ), "overriden min-max dict is not valid!"
        self.min_max_scores = override_min_max_scores

    @tf.function
    def call(self, inputs):
        # Ensure that the min-max scores are set
        assert (
            self.min_max_scores is not None
        ), "the model cannot be called before setting the min-max scores!"

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"]
        flow_packets = inputs["flow_packets"]
        flow_packet_size = inputs["flow_packet_size"]
        link_capacity = inputs["link_capacity"]
        link_to_path = inputs["link_to_path"]
        path_to_link = inputs["path_to_link"]
        # Maks - nodes
        nodes = inputs["nodes"]
        link_to_node = inputs["link_to_node"]
        # Przez jaki path przechodzi jaki node
        link_device_type = inputs["link_device_type"]
        link_device_type = tf.one_hot(link_device_type, depth=1)
        node_to_link = inputs["node_to_link"]
        node_to_path = inputs["node_to_path"]
        path_to_node = inputs["path_to_node"]

        # Zbieramy ruch dla każdego path'a NA LINKU według średniego ABV flow'a
        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])

        path_gather_packets = tf.gather(flow_packets, path_to_link[:, :, 0])

        link_packets = tf.math.reduce_sum(path_gather_packets, axis=1)

        # Zliczamy zsumowany load na linku
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)

        # Initialize the initial hidden state for paths
        path_state = self.path_embedding(
            tf.concat(
                [
                    (flow_traffic - self.min_max_scores["flow_traffic"][0])
                    * self.min_max_scores["flow_traffic"][1],
                    (flow_packets - self.min_max_scores["flow_packets"][0])
                    * self.min_max_scores["flow_packets"][1],
                    (flow_packet_size - self.min_max_scores["flow_packet_size"][0])
                    * self.min_max_scores["flow_packet_size"][1],
                ],
                axis=1,
            )
        )

        # Initialize the initial hidden state for links
        link_state = self.link_embedding(
            tf.concat(
                [
                    (link_capacity - self.min_max_scores["link_capacity"][0])
                    * self.min_max_scores["link_capacity"][1],
                    load,
                    link_device_type,
                ],
                axis=1,
            ),
        )

        device_link_gather = tf.gather(link_state, link_to_node[:, :], batch_dims=0)

        device_link_sum = tf.math.reduce_sum(device_link_gather, axis=1)

        device_link_mean = tf.math.reduce_mean(device_link_sum, axis=1)

        device_link_mean = tf.expand_dims(device_link_mean, axis=1)

        devices_encoded = tf.squeeze(tf.one_hot(nodes, depth=1))

        devices_encoded = tf.expand_dims(devices_encoded, axis=1)

        # Initial hidden state of devices
        device_state = self.device_embedding(
            tf.concat(
                # device_type
                # sumofalltheNodesState
                [devices_encoded, device_link_mean],
                axis=1,
            ),
        )

        # Iterate t times doing the message passing
        for _ in range(self.iterations):

            ####################
            #  LINKS TO PATH   #
            ####################
            link_gather_for_path = tf.gather(
                link_state, link_to_path, name="LinkToPath"
            )
            previous_path_state = path_state
            # stan node'a dodaje do path update jako drugi feature
            device_gather_for_path = tf.gather(device_state, node_to_path)

            link_device_gathered = (
                tf.concat(
                    [
                        tf.expand_dims(link_gather_for_path, axis=2),
                        tf.expand_dims(device_gather_for_path, axis=2),
                    ],
                    axis=2,
                ),
            )

            link_device_sum = tf.math.reduce_sum(link_device_gathered[0], axis=2)

            path_state_sequence, path_state = self.path_update(
                link_device_sum, initial_state=path_state
            )
            # We select the element in path_state_sequence so that it corresponds to the state before the link was considered
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            ###################
            #   PATH TO LINK  #
            ###################

            path_gather = tf.gather_nd(
                path_state_sequence, path_to_link, name="PathToRLink"
            )
            path_sum_link = tf.math.reduce_sum(path_gather, axis=1)
            link_state, _ = self.link_update(path_sum_link, states=link_state)

            ####################
            #  NODES Update    # node'y w zależności od linków
            ####################
            # test = tf.expand_dims(path_to_node, axis=2)
            node_gather = tf.gather_nd(
                path_state_sequence, path_to_node, name="PathToRNode"
            )
            path_sum_device = tf.math.reduce_sum(node_gather, axis=1)
            device_state, _ = self.device_update(path_sum_device, states=device_state)

            # device update
        capacity_gather = tf.gather(link_capacity, link_to_path)
        jitter = self.readout_path(
            tf.concat(
                    [
                        capacity_gather,
                        tf.expand_dims(tf.math.reduce_max(path_state_sequence[:, 1:], axis=2), axis=2),
                        tf.expand_dims(tf.math.reduce_sum(link_gather_for_path, axis=2), axis=2),
                    ],
                    axis=2,
                ),
            )
        jitter_final = tf.math.reduce_sum(jitter, axis=1)
        return jitter_final