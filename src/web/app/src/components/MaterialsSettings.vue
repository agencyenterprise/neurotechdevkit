<template>
  <div class="accordion accordion-flush" id="materialsAccordion">
    <div class="accordion-item" id="{{material}}" hidden>
      <h2 class="accordion-header" id="flush-{{loop.index}}">
        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
          data-bs-target="#flush-collapse{{loop.index}}" aria-expanded="false"
          aria-controls="flush-collapse{{loop.index}}">
          {{ material_label }}
        </button>
      </h2>
      <div id="flush-collapse{{loop.index}}" class="accordion-collapse collapse"
        aria-labelledby="flush-heading{{loop.index}}" data-bs-parent="#materialsAccordion">
        <div class="accordion-body">
          <form name="flush-collapse{{loop.index}}">
            <fieldset>
              <div class="mb-3">
                <label for="{{material}}Vp" data-bs-toggle="tooltip" data-bs-placement="right"
                  title="The speed of sound (in m/s)">VP</label>
                <input type="number" class="form-control" name="{{material}}Vp" id="{{material}}Vp"
                  onchange="valueChanged()" required />
              </div>
              <div class="mb-3">
                <label for="{{material}}Rho" data-bs-toggle="tooltip" data-bs-placement="right"
                  title="The mass density (in kg/m³)">RHO</label>
                <input type="number" class="form-control" name="{{material}}Rho" id="{{material}}Rho"
                  onchange="valueChanged()" required />
              </div>
              <div class="mb-3">
                <label for="{{material}}Alpha" data-bs-toggle="tooltip" data-bs-placement="right"
                  title="The absorption (in dB/cm)">Alpha</label>
                <input type="number" step="any" class="form-control" name="{{material}}Alpha" id="{{material}}Alpha"
                  onchange="valueChanged()" required />
              </div>
              <div class="mb-3">
                <label for="{{material}}RenderColor" class="form-label" data-bs-toggle="tooltip" data-bs-placement="right"
                  title="The color used when rendering this material in the scenario layout plot">Color</label>
                <input type="color" class="form-control form-control-color" name="{{material}}RenderColor"
                  id="{{material}}RenderColor" title="Choose your color" onchange="valueChanged()" />
              </div>
            </fieldset>
          </form>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
}
</script>

<style scoped></style>
