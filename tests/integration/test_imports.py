def test_public_package_imports():
    import tpso
    import tpso.contexts
    import tpso.pipelines
    import tpso.runner

    assert tpso.__version__ == "0.1.0"
