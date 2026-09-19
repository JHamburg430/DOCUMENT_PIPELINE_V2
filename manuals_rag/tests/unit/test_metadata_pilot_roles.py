from manuals_rag_parsers import metadata as m


def test_publication_code_is_not_a_part_number():
    text = 'Copyright 2024 EXAMPLE. Q47GB Printed in Japan'
    extraction = m.ScopedMetadataExtraction.model_validate({'entities':[dict(
        value='Q47GB',kind='part_number',relation='mentioned',source_quote=text,confidence=.95)]})
    assert m._ground_scoped_candidates(extraction,[m.MetadataSourceSegment(text,12,12)]) == []


def test_accessory_target_is_not_promoted_to_primary_product():
    text = 'Heavy-Duty Protective Column for ZX-R Light Curtains'
    extraction = m.ScopedMetadataExtraction.model_validate({'entities':[dict(
        value='ZX-R',kind='product_model',relation='primary_product',source_quote=text,confidence=.95)]})
    claims=m._ground_scoped_candidates(extraction,[m.MetadataSourceSegment(text,2,2)])
    assert len(claims)==1 and claims[0]['relation']=='mentioned'


def test_verified_cover_controller_precedes_head_table_primary_label():
    title='ZX-8000 Series Configuration Manual'
    base=m.DocumentMetadata(manufacturer='Example',companies=[],product_family=None,product_model=None,
        product_families=[],product_models=[],devices=[],part_numbers=[],protocol_terms=[],settings=[],
        parameters=[],menu_labels=[],document_topics=[],title=title,document_kind=m.DocumentKind('manual'),
        revision_date=None,effective_date=None)
    common=dict(kind='product_model',grounded=True,verification_status='confirmed',confidence=.95,subject=None,section_path=[])
    claims=[dict(common,value='ZX-015',relation='primary_product',source_quote='Model ZX-015',page_from=3,page_to=3),
            dict(common,value='ZX-8000',relation='mentioned',source_quote=title,page_from=1,page_to=1,source_method='opening_title_candidate')]
    result=m._materialize_verified_metadata(base,title,None,claims,[m.MetadataSourceSegment(title,1,1),m.MetadataSourceSegment('Model ZX-015',3,3)])
    assert result.product_model=='ZX-8000'
    assert 'ZX-015' in result.routing_product_models
