def test_import_package():
    import statmerge
    assert hasattr(statmerge,'run_gui')
