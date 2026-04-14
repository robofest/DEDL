import os
import ray
import numpy as np
import random

# ==== DATA ====
x = np.random.rand(10000, 20).astype(np.float32)
y = np.sum(x, axis=1).astype(np.float32)

# ==== GENOME ====
def create_individual():
    return {
        "units1": random.choice([32, 64, 128]),
        "units2": random.choice([32, 64, 128]),
        "lr": random.choice([1e-2, 5e-3, 1e-3, 1e-4])
    }

def mutate(ind):
    return {
        "units1": random.choice([32, 64, 128]),
        "units2": random.choice([32, 64, 128]),
        "lr": ind["lr"]
    }

def crossover(p1, p2):
    return {
        "units1": random.choice([p1["units1"], p2["units1"]]),
        "units2": random.choice([p1["units2"], p2["units2"]]),
        "lr": random.choice([p1["lr"], p2["lr"]])
    }

# ==== RAY GPU WORKER ====
@ray.remote(num_gpus=1)
class GPUWorker:
    def __init__(self, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import tensorflow as tf
        self.tf = tf

        # Enable memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except:
                pass

    def evaluate(self, individual, seed):
        tf = self.tf

        # Deterministic
        tf.keras.utils.set_random_seed(seed)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(individual["units1"], activation='relu'),
            tf.keras.layers.Dense(individual["units2"], activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=individual["lr"]),
            loss='mse'
        )

        model.fit(x, y, epochs=3, batch_size=128, verbose=0)
        loss = model.evaluate(x, y, verbose=0)

        tf.keras.backend.clear_session()
        return loss


# ==== MAIN GA LOOP ====
if __name__ == "__main__":
    ray.init()

    NUM_GPUS = 8
    POP_SIZE = 8
    GENERATIONS = 5

    # Create workers (1 per GPU)
    workers = [GPUWorker.remote(i) for i in range(NUM_GPUS)]

    # Initialize population
    population = [create_individual() for _ in range(POP_SIZE)]

    # Track global best
    global_best_individual = None
    global_best_loss = float("inf")

    for gen in range(GENERATIONS):
        print(f"\n=== Generation {gen} ===")

        # ---- parallel evaluation ----
        futures = []
        for i, individual in enumerate(population):
            worker = workers[i % NUM_GPUS]
            seed = 42 + gen * 100 + i
            futures.append(worker.evaluate.remote(individual, seed))

        losses = ray.get(futures)

        # Print all individuals
        for i, loss in enumerate(losses):
            print(f"{population[i]} → Loss: {loss:.6f}")

        # ---- sort by loss (ascending) ----
        sorted_pairs = sorted(zip(losses, population), key=lambda t: t[0])
        population = [ind for _, ind in sorted_pairs]

        # Best in this generation
        best_loss, best_individual = sorted_pairs[0]

        print(f"Best (gen): {best_individual} | Loss: {best_loss:.6f}")

        # ---- update global best ----
        if best_loss < global_best_loss:
            global_best_loss = best_loss
            global_best_individual = best_individual

        print(f"Best (global): {global_best_individual} | Loss: {global_best_loss:.6f}")

        # ---- elitism ----
        new_population = population[:2]

        # ---- reproduction ----
        while len(new_population) < POP_SIZE:
            p1, p2 = random.sample(population[:4], 2)
            child = mutate(crossover(p1, p2))
            new_population.append(child)

        population = new_population

    print("\n=== FINAL RESULT ===")
    print(f"Best individual found: {global_best_individual}")
    print(f"Best loss: {global_best_loss:.6f}")

    ray.shutdown()