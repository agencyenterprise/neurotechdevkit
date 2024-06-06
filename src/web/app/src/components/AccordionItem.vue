<template>
  <div class="accordion-item">
    <h2 class="accordion-header" @click="$emit('toggle', index)">
      {{ title }}
      <font-awesome-icon :icon="icon" />
    </h2>
    <div v-show="isOpen" class="accordion-content">
      <!-- This slot will be replaced by the content passed to the component -->
      <slot></slot>
    </div>
  </div>
</template>

<script>
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome';

export default {
  components: {
    FontAwesomeIcon
  },
  props: {
    title: {
      type: String,
      required: true
    },
    index: {
      type: Number,
      required: true
    },
    isOpen: {
      type: Boolean,
      required: true
    }
  },
  computed: {
    icon() {
      return this.isOpen ? 'chevron-up' : 'chevron-down';
    }
  }
}
</script>

<style scoped>
.accordion-item {
  margin-bottom: 10px;
}

.accordion-header {
  background-color: #f5f5f5;
  padding: 10px 15px;
  cursor: pointer;
  border-bottom: 1px solid #ddd;
  display: flex;
  justify-content: space-between;
  align-items: center;
  user-select: none;
}

.accordion-content {
  padding: 10px 15px;
}

.accordion-content>* {
  margin-bottom: 0;
}
</style>