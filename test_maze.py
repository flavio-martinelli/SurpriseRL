from unittest import TestCase
import numpy as np
from maze import Maze


class TestMaze(TestCase):
    def test_init(self):
        mz = Maze(size=5)

        self.assertEqual(mz.maze_dimensionality, 1)
        self.assertEqual(mz.current_pos.shape, (1,))
        self.assertEqual(mz.current_pos[0].shape, ())

    def test_get_random_position(self):
        mz = Maze(size=(5, 6, 7))

        posz = []
        for i in range(100):
            posz.append(mz.get_random_position())

        posz = np.array(posz)

        self.assertLess(posz.max(axis=0)[0], 5)
        self.assertLess(posz.max(axis=0)[1], 6)
        self.assertLess(posz.max(axis=0)[2], 7)

        mz = Maze(size=12)

        posz = []
        for i in range(100):
            posz.append(mz.get_random_position())

        posz = np.array(posz)

        self.assertLess(posz.max(axis=0), 12)

    def test_generate_train_max_spikes(self):
        mz = Maze((3, 4))
        mz.set_spike_train_params(20)
        spikes = mz.generate_spike_train((1, 1))

        self.assertEqual(spikes[:, 1, 1].sum(), 20)

    def test_generate_train_shape(self):
        mz = Maze((3, 4))
        mz.set_spike_train_params(20)
        spikes = mz.generate_spike_train((1, 1))

        self.assertEqual(spikes.shape, (20, 3, 4))

    def test_generate_train_max_weirdshapes(self):
        mz1 = Maze((3, 4, 2, 2))
        mz1.set_spike_train_params(2)
        spikes = mz1.generate_spike_train((2, 3, 0, 1))
        self.assertEqual(spikes.shape, (2, 3, 4, 2, 2))

        mz2 = Maze(10)
        mz2.set_spike_train_params(200)
        spikes = mz2.generate_spike_train([5])
        self.assertEqual(spikes.shape, (200, 10))

    def test_generate_possible_moves(self):
        mz = Maze(size=(4, 7))

        d = mz._generate_possible_moves(dims=4, symmetric=True)

        self.assertEqual(len(d.keys()), 8)
        self.assert_(np.array_equal(d[0], np.array([1,0])))
        self.assert_(np.array_equal(d[1], np.array([3,0])))
        self.assert_(np.array_equal(d[2], np.array([0,1])))
        self.assert_(np.array_equal(d[3], np.array([0,6])))

        for i in range(100):
            d = mz._generate_possible_moves(dims=4, symmetric=True)
            self.assertLess(d[4][0], 4)
            self.assertLess(d[4][1], 7)
            self.assertGreater(d[4][0], -4)
            self.assertGreater(d[4][1], -7)
            self.assert_(np.array_equal(d[5], -d[4]))

        d = mz._generate_possible_moves(dims=1, symmetric=True)

        self.assertEqual(len(d.keys()), 2)
        self.assert_(np.array_equal(d[0], np.array([1, 0])))
        self.assert_(np.array_equal(d[1], np.array([3, 0])))

        d = mz._generate_possible_moves(dims=16, symmetric=False)
        self.assertEqual(len(d.keys()), 16)

