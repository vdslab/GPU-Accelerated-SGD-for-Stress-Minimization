use baseline_sparse_sgd_non_gpu::{graph::Graph, preprocessing::*};
use rand::{rngs::StdRng, SeedableRng};
#[test]
fn cpu_expected_path_and_inf() {
    let graph = Graph::try_from_edges(4, &[(0, 1), (1, 2)]).unwrap();
    assert_eq!(graph.shortest_path_distances(0), vec![0, 1, 2, INF]);
    assert_eq!(graph.shortest_path_distances(2), vec![2, 1, 0, INF]);
    assert_eq!(graph.shortest_path_distances(3), vec![INF, INF, INF, 0]);
}
#[test]
fn cpu_fixtures_and_csr() {
    for graph in fixtures().unwrap() {
        verify_all_sources(&graph, &mut CpuBfs(&graph)).unwrap();
        let csr = Csr::from_graph(&graph).unwrap();
        assert_eq!(csr.columns.len(), 2 * graph.edge_size);
        for u in 0..graph.node_size {
            assert_eq!(
                csr.columns[csr.rows[u] as usize..csr.rows[u + 1] as usize],
                graph
                    .neighbors(u)
                    .iter()
                    .map(|&v| v as u32)
                    .collect::<Vec<_>>()
            );
        }
    }
}
#[test]
fn backend_pipeline_preserves_baseline_rng_and_sgd() {
    let graph = Graph::try_from_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]).unwrap();
    for seed in [0, 1, 42, 999] {
        let mut r1 = StdRng::seed_from_u64(seed);
        let mut r2 = r1.clone();
        let old = graph
            .prepare_sgd_params(15, 0.1, 200, true, &mut r1)
            .unwrap();
        let new = prepare(&graph, &mut CpuBfs(&graph), 200, 15, 0.1, true, &mut r2).unwrap();
        assert_eq!(old, new.params);
        let a = baseline_sparse_sgd_non_gpu::algorithm::execute_sgd(old, &mut r1);
        let b = baseline_sparse_sgd_non_gpu::algorithm::execute_sgd(new.params, &mut r2);
        assert_eq!(a, b);
        assert_eq!(full_stress(&graph, &a), full_stress(&graph, &b));
    }
}
#[test]
fn malformed_csr_rejected() {
    for csr in [
        Csr {
            rows: vec![0, 2],
            columns: vec![0],
        },
        Csr {
            rows: vec![1, 1],
            columns: vec![0],
        },
        Csr {
            rows: vec![0, 1],
            columns: vec![1],
        },
    ] {
        assert!(csr.validate().is_err());
    }
}
#[test]
fn wrong_backend_fails_validation() {
    let graph = Graph::try_from_edges(3, &[(0, 1), (1, 2)]).unwrap();
    struct Bad;
    impl DistanceBackend for Bad {
        fn distances(&mut self, _: usize) -> anyhow::Result<Vec<u32>> {
            Ok(vec![0])
        }
    }
    assert!(verify_all_sources(&graph, &mut Bad).is_err());
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_all_sources_and_pivot_pipeline() {
    use baseline_sparse_sgd_non_gpu::cuda::CudaBfs;
    for graph in fixtures().unwrap() {
        let csr = Csr::from_graph(&graph).unwrap();
        let mut cuda = CudaBfs::new(&csr).unwrap(); // Fail if GPU unavailable; never silently skip.
        verify_all_sources(&graph, &mut cuda).unwrap();
        // Repeated calls, varied sources: catch stale distance/frontier state.
        verify_all_sources(&graph, &mut cuda).unwrap();
        if graph.node_size > 1 && graph.ensure_connected().is_ok() {
            for seed in [0, 42] {
                let mut cpu_rng = StdRng::seed_from_u64(seed);
                let mut gpu_rng = cpu_rng.clone();
                let cpu = prepare(
                    &graph,
                    &mut CpuBfs(&graph),
                    graph.node_size.min(8),
                    15,
                    0.1,
                    true,
                    &mut cpu_rng,
                )
                .unwrap();
                let gpu = prepare(
                    &graph,
                    &mut cuda,
                    graph.node_size.min(8),
                    15,
                    0.1,
                    true,
                    &mut gpu_rng,
                )
                .unwrap();
                verify_prepared(&cpu, &gpu).unwrap();
                let a =
                    baseline_sparse_sgd_non_gpu::algorithm::execute_sgd(cpu.params, &mut cpu_rng);
                let b =
                    baseline_sparse_sgd_non_gpu::algorithm::execute_sgd(gpu.params, &mut gpu_rng);
                assert_eq!(a, b);
                assert_eq!(full_stress(&graph, &a), full_stress(&graph, &b));
            }
        }
    }
}

#[test]
fn fixed_branch_and_cycle_matrices() {
    let graphs = fixtures().unwrap();
    let expected = [
        vec![
            vec![0, 1, 2, 2, 3, 3],
            vec![1, 0, 1, 1, 2, 2],
            vec![2, 1, 0, 2, 3, 3],
            vec![2, 1, 2, 0, 1, 1],
            vec![3, 2, 3, 1, 0, 2],
            vec![3, 2, 3, 1, 2, 0],
        ],
        vec![
            vec![0, 1, 2, 1, 2],
            vec![1, 0, 1, 2, 3],
            vec![2, 1, 0, 1, 2],
            vec![1, 2, 1, 0, 1],
            vec![2, 3, 2, 1, 0],
        ],
    ];
    for (graph, matrix) in graphs[1..3].iter().zip(expected) {
        for (source, row) in matrix.into_iter().enumerate() {
            assert_eq!(graph.shortest_path_distances(source), row);
        }
    }
}

#[test]
fn golden_hashes_from_unmodified_cpu_reference() {
    // Generated from graph.rs at commit 419c89d01cb229b498cb1a5ce87c544a354884bd.
    // Pin RNG consumption, pivot order, all distance rows, weights, coordinates, etas.
    let golden = [
        (
            0,
            [
                9331983234788447652,
                12777333201042154693,
                17829010163290710905,
                3092003535333152786,
                6396460311482570312,
            ],
        ),
        (
            1,
            [
                13928905853220542884,
                1140848119288923205,
                17829010163290710905,
                12781450557734364035,
                6396460311482570312,
            ],
        ),
        (
            42,
            [
                925268474280206052,
                2977502081295217605,
                17829010163290710905,
                17099927555727156036,
                6396460311482570312,
            ],
        ),
        (
            999,
            [
                17104909406310284100,
                6845286157140875141,
                17829010163290710905,
                17818847101966441668,
                6396460311482570312,
            ],
        ),
    ];
    let graph = Graph::try_from_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]).unwrap();
    for (seed, expected) in golden {
        let actual = prepare(
            &graph,
            &mut CpuBfs(&graph),
            200,
            15,
            0.1,
            true,
            &mut StdRng::seed_from_u64(seed),
        )
        .unwrap();
        assert_eq!(hashes(&actual), expected);
    }
}
