import { createApp } from "vue";
import App from "./App.vue";
import { library } from "@fortawesome/fontawesome-svg-core";
import { faChevronUp, faChevronDown } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/vue-fontawesome";
import "bootstrap/dist/css/bootstrap.css";
import "bootstrap/dist/js/bootstrap.js";
import { Tooltip } from "bootstrap";
import store from "./store/index";

library.add(faChevronDown, faChevronUp);

const app = createApp(App);

// Define a global custom directive called 'tooltip'
app.directive("tooltip", {
  // When the bound element is mounted, and whenever it updates, initialize/update the tooltip
  mounted(el, binding) {
    // Create a new Tooltip instance for the element
    const tooltipTrigger = new Tooltip(el, {
      title: binding.value || el.getAttribute("title"),
      placement: binding.arg || "top",
      trigger: "hover focus",
      container: "body", // Append the tooltip to the body to prevent layout issues
    });
    // Store the tooltip instance on the element for future updates
    el._tooltip = tooltipTrigger;
  },
  updated(el, binding) {
    if (el._tooltip) {
      // Update the tooltip content and options
      el._tooltip.setContent({
        ".tooltip-inner": binding.value || el.getAttribute("title"),
      });
      if (binding.arg) {
        el._tooltip.updateAttachment(binding.arg);
      }
    }
  },
  beforeUnmount(el) {
    if (el._tooltip) {
      // Destroy the tooltip instance when the directive is unbound
      el._tooltip.dispose();
      delete el._tooltip;
    }
  },
});

app.use(store);
app.component("font-awesome-icon", FontAwesomeIcon);
app.mount("#app");
