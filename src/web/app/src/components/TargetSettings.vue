<template>
  <form id="targetForm">
    <fieldset>
      <div class="mb-3 row">
        <div class="col">
          <label for="centerX" data-bs-toggle="tooltip" data-bs-placement="right"
            title="The location of the center of the target (in meters)">Center X</label>
          <input type="number" step="any" class="form-control" name="centerX" id="centerX" onchange="valueChanged()"
            required />
        </div>
        <div class="col">
          <label for="centerY" data-bs-toggle="tooltip" data-bs-placement="right"
            title="The location of the center of the target (in meters)">Center Y</label>
          <input type="number" step="any" class="form-control" id="centerY" name="centerY" onchange="valueChanged()"
            required />
        </div>
        <div class="col" data-3d-only="true" hidden>
          <label for="centerZ" data-bs-toggle="tooltip" data-bs-placement="right"
            title="The location of the center of the target (in meters)">Center Z</label>
          <input type="number" step="any" class="form-control" id="centerZ" name="centerZ" onchange="valueChanged()"
            data-3d-input-only="true" />
        </div>
      </div>
      <div class="mb-3">
        <label for="radius" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The radius of the target (in meters)">Radius</label>
        <input type="number" step="any" class="form-control" name="radius" id="radius" onchange="valueChanged()"
          required />
      </div>
    </fieldset>
  </form>
</template>

<script>
export default {
  name: 'HelloWorld',
  props: {
    msg: String
  }
}
</script>

<style scoped></style>
