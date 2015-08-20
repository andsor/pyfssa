{%- extends 'rst.tpl' -%}

{% block input %}
{%- if cell.source.strip() -%}
.. code-block:: python

{{ cell.source | indent}}
{% endif -%}
{% endblock input %}
